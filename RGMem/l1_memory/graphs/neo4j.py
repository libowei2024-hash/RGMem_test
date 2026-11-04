#l1_memory/graphs/neo4j.py
import logging
import json
import asyncio
from datetime import datetime
import pytz
from typing import Any, Dict, List, Optional
from l1_memory.graphs.value_conclusions import ValueConclusionProcessor
from config import settings
from l1_memory.graphs.inference import InferenceProcessor
from l1_memory.graphs.summary import SummaryProcessor
from l1_memory.graphs.tools import (
    RELATIONS_TOOL,
    EXTRACT_CORE_ENTITIES_TOOL,
    SEARCH_ENTITY_EXTRACTION_TOOL,
)
from l1_memory.graphs.utils import EXTRACT_RELATIONS_PROMPT, EXTRACT_ENTITIES_PROMPT, SEARCH_ENTITY_EXTRACTION_PROMPT
from l1_memory.utils.factory import EmbedderFactory, LlmFactory

try:
    from langchain_neo4j import Neo4jGraph
except ImportError:
    raise ImportError("langchain_neo4j is not installed. Please install it using pip install langchain-neo4j")
from l1_memory.graphs.graph_retrieval import GraphRetriever
logger = logging.getLogger(__name__)
ENTITY_PATH_EXTRACTION_MAX_ATTEMPTS = 3
ENTITY_PATH_SCHEMA_RETRY_INSTRUCTION = """
Your previous extract_core_entities tool call violated the required schema.
Return entity_paths strictly as an array of objects, and every object must contain
a path array: {"entity_paths": [{"path": [{"name": "User", "type": "abstract"}]}]}.
Each entity type must be exactly either "abstract" or "event".
Do not return a path as a bare array.
""".strip()


def validate_entity_paths(entity_paths: Any) -> List[Dict[str, List[Dict[str, str]]]]:
    """Validate and normalize only the documented entity-path tool schema."""
    if not isinstance(entity_paths, list):
        raise TypeError("entity_paths must be a list")
    if not entity_paths:
        raise ValueError("entity_paths must contain at least one path object")

    cleaned_paths = []
    for path_index, path_obj in enumerate(entity_paths):
        if not isinstance(path_obj, dict):
            raise TypeError(
                f"entity_paths[{path_index}] must be an object containing 'path'"
            )
        if "path" not in path_obj:
            raise ValueError(f"entity_paths[{path_index}] is missing required 'path'")
        raw_path = path_obj["path"]

        if not isinstance(raw_path, list):
            raise TypeError(f"entity_paths[{path_index}].path must be a list")
        if not raw_path:
            raise ValueError(f"entity_paths[{path_index}].path must not be empty")

        cleaned_paths.append(
            {
                "path": [
                    {
                        "name": entity["name"].lower().replace(" ", "_"),
                        "type": entity["type"],
                    }
                    for entity in raw_path
                ]
            }
        )
    return cleaned_paths


class MemoryGraph:
    def __init__(self, config):
        self.config = config
        logger.info(f"正在初始化 MemoryGraph，尝试连接到 Neo4j: {self.config.graph_store.config.url}")
        self.graph = Neo4jGraph(
            url=self.config.graph_store.config.url,
            username=self.config.graph_store.config.username,
            password=self.config.graph_store.config.password,
            database=self.config.graph_store.config.database,
            refresh_schema=False,  # Avoids unnecessary schema checks on init
            driver_config={"notifications_min_severity": "OFF"},
        )
        logger.info("Neo4j 连接成功。")
        self.embedding_model = EmbedderFactory.create(self.config.embedder.provider, self.config.embedder.config)
        self.node_label = ""
        try:
            self.graph.query("CREATE INDEX event_user_id_index IF NOT EXISTS FOR (n:event) ON (n.user_id)")
            self.graph.query("CREATE INDEX abstract_user_id_index IF NOT EXISTS FOR (n:abstract) ON (n.user_id)")
            logger.info("成功为 :event 和 :abstract 标签创建或确认了 user_id 的 B-Tree 索引。")
        except Exception as e:
            logger.error(f"创建 B-Tree 索引时发生错误: {e}", exc_info=True)
            pass

        self.llm_provider = self.config.llm.provider
        self.llm = LlmFactory.create(self.llm_provider, self.config.llm.config)
        self.value_conclusion_processor = ValueConclusionProcessor(llm_client=self.llm)

        self.threshold = settings.retrieval_similar_threshold
        self.node_similar_threshold = settings.node_similar_threshold

        self.inference_processor = InferenceProcessor(graph_client=self.graph, llm_client=self.llm)
        self.summary_processor = SummaryProcessor(graph_client=self.graph, llm_client=self.llm)
        self.retriever = GraphRetriever(
            graph_client=self.graph,
            llm_client=self.llm,
            embedding_model=self.embedding_model,
            config=self.config
        )

        self.inference_threshold = getattr(settings, "inference_threshold", 5)
        self.summary_threshold = getattr(settings, "summary_threshold", 20)
        logger.info(f"触发器阈值已设置: Inference={self.inference_threshold}, Summary={self.summary_threshold}")

    def add(self, input_data: Dict[str, Any], filters: Dict[str, Any]):

        text_to_analyze = f"Core Fact: {input_data['core_fact']}\n\nConclusions: {json.dumps(input_data.get('conclusions', []), ensure_ascii=False)}"
        entity_paths = self._retrieve_nodes_from_data(text_to_analyze, filters)

        if not entity_paths:
            logger.warning("实体提取未能从文本中返回任何路径，处理中止。")
            return {"write_results": {"added_entities": []}}

        # 从路径列表中构建一个扁平的 entity_type_map，以便下游方法快速查找类型
        entity_type_map = {
            entity['name']: entity['type']
            for path_obj in entity_paths
            for entity in path_obj.get('path', [])
        }

        # 步骤2: LLM基于路径上下文提取关系，并由代码进行校准
        relationships_to_add = self._establish_nodes_relations_from_data(text_to_analyze, filters, entity_paths)

        # 步骤3: [核心] 在所有后续处理之前，立即进行关系去重
        unique_relationships = {}
        for rel in relationships_to_add:
            # 使用 (source, destination) 作为唯一键
            key = (rel["source"], rel["destination"])
            if key not in unique_relationships:
                unique_relationships[key] = rel

        final_unique_relationships = list(unique_relationships.values())

        logger.info(
            f"关系去重：原始关系数量 {len(relationships_to_add)}，去重后唯一关系数量 {len(final_unique_relationships)}。")

        # 步骤4: 使用【去重后】的关系列表进行高价值结论的匹配
        # 只筛选 event 类型的关系作为结论匹配的候选池
        event_relationships_for_conclusions = [
            rel for rel in final_unique_relationships if rel.get('relationship_type') == 'event'
        ]

        conclusion_map = {}
        if input_data.get("conclusions"):
            conclusion_map = self.value_conclusion_processor.match_conclusions_to_triplets(
                conclusions_data=input_data,
                all_triplets=event_relationships_for_conclusions  # 使用去重后的列表
            )

        # 步骤5: 使用【去重后】的关系列表和匹配好的结论，执行最终的写入操作
        write_results_list = self._add_entities(
            relationships_to_add=final_unique_relationships,  # 使用去重后的列表
            filters=filters,
            entity_type_map=entity_type_map,
            input_data=input_data,
            conclusion_map=conclusion_map
        )

        return {"write_results": {"added_entities": write_results_list}}

    def _add_entities(self, relationships_to_add: List[Dict], filters: Dict, entity_type_map: Dict,
                      input_data: Dict, conclusion_map: Dict[tuple, list]):
        results = []
        event_timestamp = input_data.get("created_at", datetime.now(pytz.utc).isoformat())

        # 为所有需要嵌入的唯一实体名称创建嵌入向量，以减少重复计算
        unique_entities = {rel["source"] for rel in relationships_to_add} | {rel["destination"] for rel in
                                                                             relationships_to_add}
        embedding_cache = {name: self.embedding_model.embed(name) for name in unique_entities}

        for item in relationships_to_add:
            original_source_name = item["source"]
            original_dest_name = item["destination"]
            source_type = entity_type_map.get(original_source_name, 'abstract')
            destination_type = entity_type_map.get(original_dest_name, 'event')
            source_embedding = embedding_cache.get(original_source_name)
            dest_embedding = embedding_cache.get(original_dest_name)

            # 步骤 a: 相似度合并 (带类型约束)
            final_source_name, final_source_type = self._search_and_get_node_name(
                source_embedding, filters, original_source_name, source_type, is_source=True, input_data=input_data
            )
            final_dest_name, final_dest_type = self._search_and_get_node_name(
                dest_embedding, filters, original_dest_name, destination_type, is_source=False, input_data=input_data
            )

            # 步骤 b: 准备写入内容
            relation_history_entry = {
                "relationship_content": item.get("relationship"),
                "timestamp": event_timestamp,
                "memory_id": input_data["memory_id"]
            }
            event_history_entry = {
                "timestamp": event_timestamp,
                "memory_id": input_data["memory_id"]
            }

            triplet_key = (original_source_name, item["relationship"], original_dest_name)
            conclusions_for_this_triplet = conclusion_map.get(triplet_key, [])

            l1_conclusions_str_list = [json.dumps(c, ensure_ascii=False) for c in conclusions_for_this_triplet]

            # 步骤 c: 日志打印 (保持不变)
            logger.info("--- [写入图谱] 准备写入以下逻辑关系 ---")
            logger.info(f"  源实体: '{final_source_name}' (类型: {final_source_type})")
            logger.info(f"  目标实体: '{final_dest_name}' (类型: {final_dest_type})")
            logger.info(f"  关系: '{item['relationship']}' (逻辑类型: {item.get('relationship_type')})")
            if l1_conclusions_str_list:
                logger.info(f"  附加结论 ({len(l1_conclusions_str_list)}条):")
                for conclusion in conclusions_for_this_triplet:
                    logger.info(f"    - {conclusion['content']}")
            logger.info("-------------------------------------------")

            # 步骤 d: 构建并执行整合的Cypher查询
            role_id_merge_str = ", role_id: $role_id" if filters.get("role_id") else ""
            source_role_id_create_str = ", source.role_id = $role_id" if filters.get("role_id") else ""
            dest_role_id_create_str = ", destination.role_id = $role_id" if filters.get("role_id") else ""

            total_increment = 1 + len(l1_conclusions_str_list)

            cypher = f'''
                            MERGE (source:`{final_source_type}` {self.node_label} {{name: $source_name, user_id: $user_id{role_id_merge_str}}})
                            ON CREATE SET source.type = $source_type, source.created = $ts, source.embedding = $source_embedding,
                                          source.community_influence_score = 0.0, 
                                          source.is_dirty = false, 
                                          source.summary = null, source.last_summary_update = null {source_role_id_create_str}
                            ON MATCH SET source.is_dirty = CASE 
                                             WHEN source.type = 'abstract' THEN coalesce(source.is_dirty, false) 
                                             ELSE source.is_dirty 
                                           END

                            MERGE (destination:`{final_dest_type}` {self.node_label} {{name: $dest_name, user_id: $user_id{role_id_merge_str}}})
                            ON CREATE SET destination.type = $dest_type, destination.created = $ts, destination.embedding = $dest_embedding,
                                          destination.history = CASE WHEN $dest_type = 'event' THEN [] ELSE null END,
                                          destination.mentions = CASE WHEN $dest_type = 'event' THEN 0 ELSE null END,
                                          destination.is_dirty = CASE WHEN $dest_type = 'abstract' THEN false ELSE null END
                                          {dest_role_id_create_str}
                            ON MATCH SET destination.is_dirty = CASE 
                                             WHEN destination.type = 'abstract' THEN coalesce(destination.is_dirty, false) 
                                             ELSE destination.is_dirty 
                                           END

                            MERGE (source)-[r:RELATED_TO]->(destination)
                            ON CREATE SET r.created = $ts, r.type = $relationship_type, r.mentions = 0, 
                                          r.l1_conclusions = [], r.history = [],
                                          r.inference = null, r.last_inference_update = null
                            ON MATCH SET r.type = $relationship_type

                            SET r.history = r.history + $relation_history_entry,
                                r.l1_conclusions = r.l1_conclusions + $l1_conclusions,

                            destination.history = CASE 
                                WHEN $dest_type = 'event' THEN coalesce(destination.history, []) + [$event_history_entry] 
                                ELSE destination.history 
                            END,

                            destination.mentions = CASE
                                WHEN $dest_type = 'event' THEN coalesce(destination.mentions, 0) + 1
                                ELSE destination.mentions
                            END,

                            r.mentions = CASE
                                WHEN $relationship_type = 'event' THEN coalesce(r.mentions, 0) + $total_increment
                                ELSE r.mentions
                            END,

                            source.community_influence_score = CASE
                                WHEN source.type = 'abstract' AND $relationship_type = 'event' 
                                THEN coalesce(source.community_influence_score, 0) + $total_increment
                                ELSE source.community_influence_score
                            END

                        RETURN elementId(source) AS source_id,
                               source.community_influence_score AS new_community_score,
                               elementId(r) AS rel_id,
                               r.mentions AS new_rel_mentions
                        '''
            params = {
                "source_name": final_source_name, "source_type": final_source_type,
                "dest_name": final_dest_name, "dest_type": final_dest_type,
                "source_embedding": source_embedding, "dest_embedding": dest_embedding,
                "user_id": filters["user_id"], "role_id": filters.get("role_id"),
                "ts": event_timestamp,
                "relation_history_entry": json.dumps(relation_history_entry, ensure_ascii=False),
                "l1_conclusions": l1_conclusions_str_list,
                "event_history_entry": json.dumps(event_history_entry, ensure_ascii=False),
                "total_increment": total_increment,
                "relationship_type": item.get("relationship_type")
            }

            result = self.graph.query(cypher, params=params)
            results.append(result)

            # 步骤 e: 触发检查 (保持不变)
            if result:
                res_data = result[0]
                if item.get("relationship_type") == 'event':
                    self.check_and_trigger_inference(
                        res_data['rel_id'],
                        res_data['new_rel_mentions'],
                        event_timestamp
                    )
                if final_source_type == 'abstract' and item.get("relationship_type") == 'event':
                    self.check_and_trigger_summary(
                        res_data['source_id'],
                        res_data['new_community_score'],
                        event_timestamp
                    )
        return results

    def check_and_trigger_inference(self, relationship_id: str, new_mentions: int, event_timestamp: str):
        if new_mentions >= self.inference_threshold:
            logger.info(
                f"Inference threshold reached for relationship {relationship_id} ({new_mentions}). Triggering async update.")
            reset_query = "MATCH ()-[r]->() WHERE elementId(r) = $id SET r.mentions = 0"
            self.graph.query(reset_query, params={"id": relationship_id})

            self.inference_processor.update_inference(relationship_id, event_timestamp)


    def check_and_trigger_summary(self, abstract_node_id: str, new_community_score: float, event_timestamp: str):
        if new_community_score >= self.summary_threshold:
            logger.info(
                f"Community summary threshold reached for node {abstract_node_id} ({new_community_score}). Triggering async update.")
            reset_query = "MATCH (n) WHERE elementId(n) = $id SET n.community_influence_score = 0"
            self.graph.query(reset_query, params={"id": abstract_node_id})

            self.summary_processor.update_community_summary(abstract_node_id, event_timestamp)

    def delete_all(self, filters):
        if filters.get("role_id"):
            cypher = f"""
            MATCH (n {self.node_label} {{user_id: $user_id, role_id: $role_id}})
            DETACH DELETE n
            """
            params = {"user_id": filters["user_id"], "role_id": filters["role_id"]}
        else:
            cypher = f"""
            MATCH (n {self.node_label} {{user_id: $user_id}})
            DETACH DELETE n
            """
            params = {"user_id": filters["user_id"]}
        self.graph.query(cypher, params=params)

    def _retrieve_nodes_from_data(self, data: str, filters: Dict) -> list:
        system_prompt = EXTRACT_ENTITIES_PROMPT
        _tools = [EXTRACT_CORE_ENTITIES_TOOL]
        tool_name = "extract_core_entities"

        for attempt in range(1, ENTITY_PATH_EXTRACTION_MAX_ATTEMPTS + 1):
            attempt_system_prompt = system_prompt
            if attempt > 1:
                attempt_system_prompt = (
                    f"{system_prompt}\n\n{ENTITY_PATH_SCHEMA_RETRY_INSTRUCTION}"
                )

            logger.debug("--- LLM Call Start: Extracting Core Entity Paths ---")
            logger.debug(f"Extraction attempt: {attempt}/{ENTITY_PATH_EXTRACTION_MAX_ATTEMPTS}")
            logger.debug(f"System Prompt:\n{attempt_system_prompt}")
            logger.debug(f"User Content:\n{data}")
            logger.debug(f"Tools Used:\n{json.dumps(_tools, indent=2, ensure_ascii=False)}")

            search_results = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": attempt_system_prompt},
                    {"role": "user", "content": data},
                ],
                tools=_tools,
                tool_choice={"type": "function", "function": {"name": tool_name}}
            )

            logger.debug(f"LLM Response (Tool Calls):\n{json.dumps(search_results, indent=2, ensure_ascii=False)}")
            logger.debug("--- LLM Call End: Extracting Core Entity Paths ---")

            try:
                tool_calls = search_results.get("tool_calls", [])
                matching_call = next(
                    (
                        tool_call
                        for tool_call in tool_calls
                        if tool_call.get("name") == tool_name
                    ),
                    None,
                )
                if matching_call is None:
                    raise ValueError(f"missing required tool call: {tool_name}")
                arguments = matching_call.get("arguments", {})
                if not isinstance(arguments, dict):
                    raise TypeError("tool-call arguments must be an object")
                entity_paths = arguments.get("entity_paths", [])
                cleaned_paths = validate_entity_paths(entity_paths)
                logger.info(
                    "成功提取到 %s 条实体路径（尝试 %s/%s）。",
                    len(cleaned_paths),
                    attempt,
                    ENTITY_PATH_EXTRACTION_MAX_ATTEMPTS,
                )
                return cleaned_paths
            except (KeyError, TypeError, ValueError) as exc:
                if attempt < ENTITY_PATH_EXTRACTION_MAX_ATTEMPTS:
                    logger.warning(
                        "实体路径未通过官方 schema 校验（尝试 %s/%s）: %s。将重新请求 LLM。",
                        attempt,
                        ENTITY_PATH_EXTRACTION_MAX_ATTEMPTS,
                        exc,
                    )
                    continue
                logger.exception(
                    "实体路径在 %s 次尝试后仍未通过官方 schema 校验: %s, search_results=%s",
                    ENTITY_PATH_EXTRACTION_MAX_ATTEMPTS,
                    exc,
                    search_results,
                )

        return []

    def _establish_nodes_relations_from_data(self, data: str, filters: Dict, entity_paths: List[Dict]) -> List[Dict]:
        # 1. [保险机制] 代码自动从路径生成确定性的 classification 关系
        expected_class_rels = set()
        for path_obj in entity_paths:
            path = path_obj.get('path', [])
            if len(path) < 2:
                continue
            for i in range(len(path) - 1):
                source = path[i]
                dest = path[i + 1]
                # 只有 abstract -> abstract 的连接才构成层级分类关系
                if source.get('type') == 'abstract' and dest.get('type') == 'abstract':
                    # User -> [Category] 的关系是 has_profile_in
                    relationship = "has_profile_in" if source.get('name', '').lower() == 'user' else "has_subclass"
                    # 使用元组 (source, rel, dest, type) 以方便放入集合去重
                    expected_class_rels.add(
                        (source['name'], relationship, dest['name'], 'classification')
                    )

        logger.info(f"[保险机制] 从实体路径中自动生成了 {len(expected_class_rels)} 条预期的分类关系。")

        # 2. [优化输入] 将路径格式化为对LLM更友好的字符串
        path_strings = []
        for i, path_obj in enumerate(entity_paths):
            path_str = " -> ".join(
                [f"{entity.get('name', '')} ({entity.get('type', '')})" for entity in path_obj.get('path', [])])
            path_strings.append(f"Path {i + 1}: {path_str}")

        paths_for_prompt = "\n".join(path_strings)
        user_content_with_paths = f"Identified Entity Paths:\n{paths_for_prompt}\n\nOriginal Text: {data}"

        # 3. 调用LLM，任务不变，但有了更好的上下文
        user_identity = f"user_id: {filters['user_id']}"
        if filters.get("role_id"):
            user_identity += f", role_id: {filters['role_id']}"

        system_content = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_identity)
        if self.config.graph_store.custom_prompt:
            custom_instruction = f"\n\n**Additional Instruction:**\n{self.config.graph_store.custom_prompt}"
            system_content += custom_instruction

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content_with_paths},
        ]
        _tools = [RELATIONS_TOOL]
        tool_name = "establish_relationships"

        logger.debug("--- LLM Call Start: Establishing Relations (Path-Augmented) ---")
        logger.debug(f"System Prompt:\n{system_content}")
        logger.debug(f"User Content:\n{user_content_with_paths}")

        extracted_entities_response = self.llm.generate_response(
            messages=messages,
            tools=_tools,
            tool_choice={"type": "function", "function": {"name": tool_name}}
        )
        logger.debug(
            f"LLM Response (Tool Calls):\n{json.dumps(extracted_entities_response, indent=2, ensure_ascii=False)}")
        logger.debug("--- LLM Call End: Establishing Relations ---")

        llm_generated_rels = []
        if extracted_entities_response.get("tool_calls"):
            tool_call = extracted_entities_response["tool_calls"][0]
            arguments = tool_call.get("arguments", {})
            thought_process = arguments.get("thought_process", "Model did not provide a thought process.")
            logger.info(
                f"--- LLM Thought Process (Relation Establishment) ---\n{thought_process}\n----------------------------------------------------")
            llm_generated_rels = arguments.get("entities", [])

        # 4. [保险机制] 合并与校准
        # 使用字典来去重并合并LLM结果和代码生成结果
        final_rels_map = {
            (rel['source'], rel['relationship'], rel['destination']): rel
            for rel in llm_generated_rels
        }

        # 将代码生成的预期关系合并进去，如果LLM漏了就补上
        for source, rel, dest, rel_type in expected_class_rels:
            # 清理名称以匹配LLM的输出风格（通常是小写下划线）
            clean_source = source.lower().replace(" ", "_")
            clean_dest = dest.lower().replace(" ", "_")
            key = (clean_source, rel, clean_dest)

            if key not in final_rels_map:
                logger.warning(f"[保险机制] LLM 漏掉了预期的分类关系: {key}。已由代码自动补全。")
                final_rels_map[key] = {
                    "source": clean_source,
                    "relationship": rel,
                    "destination": clean_dest,
                    "relationship_type": rel_type
                }

        final_relationships = list(final_rels_map.values())

        # 5. 清理实体名称中的空格等，并返回最终结果
        cleaned_relationships = self._remove_spaces_from_entities(final_relationships)

        logger.info(f"关系提取完成。最终关系数量 (LLM生成 + 代码校准): {len(cleaned_relationships)}")
        return cleaned_relationships


    def _delete_entities(self, to_be_deleted, filters):
        """Delete the entities from the graph."""
        user_id = filters["user_id"]
        role_id = filters.get("role_id", None)
        results = []

        for item in to_be_deleted:
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # Build the agent filter for the query
            agent_filter = ""
            params = {
                "source_name": source,
                "dest_name": destination,
                "user_id": user_id,
            }

            if role_id:
                agent_filter = "AND n.role_id = $role_id AND m.role_id = $role_id"
                params["role_id"] = role_id

            # Delete the specific relationship between nodes
            cypher = f"""
            MATCH (n {self.node_label} {{name: $source_name, user_id: $user_id}})
            -[r:{relationship}]->
            (m {self.node_label} {{name: $dest_name, user_id: $user_id}})
            WHERE 1=1 {agent_filter}
            DELETE r
            RETURN 
                n.name AS source,
                m.name AS target,
                type(r) AS relationship
            """

            result = self.graph.query(cypher, params=params)
            results.append(result)

        return results


    def _remove_spaces_from_entities(self, entity_list):
        for item in entity_list:
            item["source"] = item["source"].lower().replace(" ", "_")
            item["relationship"] = item["relationship"].lower().replace(" ", "_")
            item["destination"] = item["destination"].lower().replace(" ", "_")
        return entity_list

    def _search_source_node(self, source_embedding, filters, source_type):
        agent_filter = ""
        if filters.get("role_id"):
            agent_filter = "AND source_candidate.role_id = $role_id"

        cypher = f"""
                    MATCH (source_candidate {self.node_label})
                    WHERE source_candidate.embedding IS NOT NULL 
                      AND source_candidate.user_id = $user_id
                      AND source_candidate.type = $source_type 
                      {agent_filter}
                    WITH source_candidate,
                         round(vector.similarity.cosine(source_candidate.embedding, $source_embedding), 4) AS source_similarity
                    WHERE source_similarity >= $threshold
                    RETURN elementId(source_candidate) as source_candidate_id, 
                           source_candidate.name as source_candidate_name,
                           source_candidate.type as source_candidate_type,
                           source_similarity
                    ORDER BY source_similarity DESC
                    LIMIT 1
                """
        params = {
            "source_embedding": source_embedding,
            "user_id": filters["user_id"],
            "source_type": source_type,  # [新增] 将类型作为参数传入
            "threshold": self.node_similar_threshold
        }
        if filters.get("role_id"):
            params["role_id"] = filters["role_id"]

        return self.graph.query(cypher, params=params)

    def _search_destination_node(self, destination_embedding, filters, destination_type):
        agent_filter = ""
        if filters.get("role_id"):
            agent_filter = "AND destination_candidate.role_id = $role_id"

        cypher = f"""
                    MATCH (destination_candidate {self.node_label})
                    WHERE destination_candidate.embedding IS NOT NULL 
                      AND destination_candidate.user_id = $user_id 
                      AND destination_candidate.type = $destination_type
                      {agent_filter}
                    WITH destination_candidate,
                         round(vector.similarity.cosine(destination_candidate.embedding, $destination_embedding), 4) AS destination_similarity
                    WHERE destination_similarity >= $threshold
                    RETURN elementId(destination_candidate) as destination_candidate_id,
                           destination_candidate.name as destination_candidate_name,
                           destination_candidate.type as destination_candidate_type,
                           destination_similarity
                    ORDER BY destination_similarity DESC
                    LIMIT 1
                """
        params = {
            "destination_embedding": destination_embedding,
            "user_id": filters["user_id"],
            "destination_type": destination_type,  # [新增] 将类型作为参数传入
            "threshold": self.node_similar_threshold,
        }
        if filters.get("role_id"):
            params["role_id"] = filters["role_id"]

        return self.graph.query(cypher, params=params)

    def reset(self):
        """Reset the graph by clearing all nodes and relationships."""
        logger.warning("Clearing graph...")
        cypher_query = """
        MATCH (n) DETACH DELETE n
        """
        return self.graph.query(cypher_query)

    def _search_and_get_node_name(self, embedding, filters, original_name, original_type, is_source, input_data):
        """
        [修改后]
        Searches for a similar node and uses an LLM for final merge decision in ambiguous cases.
        """
        result = None
        if is_source:
            result = self._search_source_node(embedding, filters, original_type)
        else:
            result = self._search_destination_node(embedding, filters, original_type)

        if not result:
            logger.info(f"--- [相似度合并 ({original_type})] 未找到相似实体，将创建新实体: '{original_name}' ---")
            return original_name, original_type

        # 候选实体已找到，开始裁决流程
        candidate_node = result[0]
        candidate_name = candidate_node['source_candidate_name' if is_source else 'destination_candidate_name']
        candidate_type = candidate_node['source_candidate_type' if is_source else 'destination_candidate_type']
        similarity = candidate_node['source_similarity' if is_source else 'destination_similarity']

        logger.info(f"--- [相似度合并 ({original_type})] 发现候选实体 ---")
        logger.info(f"  - 原始实体: '{original_name}' (类型: {original_type})")
        logger.info(f"  - KG 中已有实体: '{candidate_name}' (类型: {candidate_type})")
        logger.info(f"  - 相似度: {similarity} (阈值: {self.node_similar_threshold})")

        # 裁决流程
        if similarity >= 0.99:
            logger.info("  - 裁决: 相似度大于0.99，直接合并。")

            logger.info("------------------------------------")
            return candidate_name, candidate_type

        # [核心修改] 当相似度在灰色地带时，调用LLM
        elif similarity >= self.node_similar_threshold:
            logger.info("  - 裁决: 相似度在灰色地带，启动LLM进行最终裁决...")

            memory_context = f"Core Fact: {input_data['core_fact']}\n\nConclusions: {json.dumps(input_data.get('conclusions', []), ensure_ascii=False)}"

            should_merge = self._llm_merge_decision(
                original_entity={'name': original_name, 'type': original_type},
                candidate_entity={'name': candidate_name, 'type': candidate_type},
                memory_context=memory_context
            )

            if should_merge:
                logger.info(f"  - 最终决定: 合并为 -> '{candidate_name}'")
                logger.info("------------------------------------")
                return candidate_name, candidate_type
            else:
                logger.info(f"  - 最终决定: 不合并，创建新实体 -> '{original_name}'")
                logger.info("------------------------------------")
                return original_name, original_type

        # 理论上不会执行到这里，因为 result 存在就意味着 similarity >= threshold
        # 但作为保险，返回原始值
        return original_name, original_type

    def _llm_merge_decision(self, original_entity: Dict, candidate_entity: Dict, memory_context: str) -> bool:
        """
        Uses an LLM to make a final decision on whether to merge two similar entities.

        Args:
            original_entity (Dict): The new entity extracted from text. e.g., {'name': '...', 'type': '...'}
            candidate_entity (Dict): The existing entity found in the KG. e.g., {'name': '...', 'type': '...'}
            memory_context (str): The raw text of the memory for context.

        Returns:
            bool: True if the LLM decides to merge, False otherwise.
        """
        prompt = f"""
        **Role**:
You are a rigorous and pragmatic Knowledge Graph Entity Linking Expert. Your core objective is to build a knowledge graph that is conceptually clear and free of redundancy.

**Core Principle: Aggressive Merging**
When making your judgment, you must prioritize the "Aggressive Merging" principle. For entities that are semantically highly overlapping or merely different phrasings of the same concept, you should merge them boldly to maintain the knowledge graph's simplicity and efficiency. Only choose not to merge when a clear and meaningful distinction exists between the two entities.

**Context**:
- **User's Memory**: "{memory_context}"
- **New Entity (A)**: Extracted from the memory:
  - name: "{original_entity['name']}"
- **Candidate Entity (B)**: A similar, existing entity found in the knowledge graph:
  - name: "{candidate_entity['name']}"
**Analysis Task & Judgment Rules**:
Your task is to determine whether Entity A and Entity B should be merged, based on the following rules.

**Rule 1: When to Merge (Semantic Equivalence)**
If any of the following conditions are met, you must decide `should_merge: true`:
*   **Synonymous or Rephrased:** Entity A and B are synonyms, near-synonyms, acronyms, or simply different phrasings/word orders of the exact same core meaning.
    *   *Example:* `family_camping` and `camping_with_family` should be merged.
*   **Highly Overlapping Scope:** Although not literally identical, the concepts' scope and meaning are practically the same in the given context and common understanding. Making a distinction between them offers little value.
    *   *Example:* In most contexts, `family_values` and `family_and_parenting_values` both point to the core value of cherishing family time and should be merged.

**Rule 2: When NOT to Merge (Clear Distinction)**
If any of the following conditions are met, you must decide `should_merge: false`:
*   **Clear Sub/Super-concept Relationship:** One entity is a clear subset, superset, or specific instance of the other. Merging them would result in a loss of valuable information granularity.
    *   *Example:* `mental_health_awareness` is a subset of `health_and_wellness` and should not be merged.
*   **Fundamentally Different Core Concepts:** Although the topics are related, the entities refer to fundamentally different types of things, activities, or concepts.
    *   *Example:* `running_for_mental_health` (a personal health practice) and `charity_race_for_mental_health` (an organized public event) are fundamentally different and should not be merged.

**The Role of Context**:
Use the "User's Memory" context to precisely understand the true meaning and scope of the New Entity (A), and to help judge if it is equivalent to Entity B *in this specific context*. The context is a tool to help determine semantic equivalence, not to find evidence of uniqueness.

**Output Instruction**:
You must respond in strict JSON format. Do not add any text before or after the JSON block.
The JSON should contain two keys: "reasoning" (a clear explanation for your decision based on the rules above) and "should_merge" (a boolean value: true or false).

---
**Output Examples**:

*   **Example of a Merge (should_merge: true)**
    ```json
    {{
      "reasoning": "Entity A 'family_camping' and Entity B 'camping_with_family' are merely different word orders for the exact same core concept of 'camping with one's family'. According to Rule 1 'Synonymous or Rephrased', they should be merged to avoid redundancy.",
      "should_merge": true
    }}
    ```

*   **Example of No Merge (should_merge: false)**
    ```json
    {{
      "reasoning": "Entity A 'running_for_mental_health' refers to an ongoing, personal health practice, while Entity B 'charity_race_for_mental_health' refers to an organized public event. Although the topics are related, their 'Fundamentally Different Core Concepts' (Rule 2) mean they should not be merged.",
      "should_merge": false
    }}
    ```
        """

        try:
            logger.info("--- [LLM Merge Decision] Calling LLM for final verdict ---")
            response_str = self.llm.generate_response(  # 变量重命名为 response_str
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            # [核心修复] 增加健壮的JSON解析逻辑
            response_data = None
            if isinstance(response_str, str):
                try:
                    # 尝试从可能包含代码块的字符串中提取JSON
                    match = __import__('re').search(r'```(?:json)?\s*([\s\S]+?)\s*```', response_str)
                    if match:
                        json_str = match.group(1)
                    else:
                        json_str = response_str
                    response_data = json.loads(json_str)
                except json.JSONDecodeError:
                    logger.warning(
                        f"[LLM Merge Decision] Failed to decode JSON from LLM string response: {response_str}. Defaulting to NOT MERGE.")
                    return False
            elif isinstance(response_str, dict):
                response_data = response_str

            # 现在，我们确保 response_data 是一个字典
            if isinstance(response_data, dict) and "should_merge" in response_data:
                decision = response_data.get("should_merge", False)
                # [健壮性增强] 确保 decision 是一个真正的布尔值
                if not isinstance(decision, bool):
                    logger.warning(
                        f"[LLM Merge Decision] 'should_merge' value is not a boolean: {decision}. Defaulting to NOT MERGE.")
                    decision = False

                reasoning = response_data.get("reasoning", "No reasoning provided.")
                logger.info(f"  - LLM Reasoning: {reasoning}")
                logger.info(f"  - LLM Verdict: {'MERGE' if decision else 'DO NOT MERGE'}")
                return decision
            else:
                logger.warning(
                    f"[LLM Merge Decision] LLM returned an invalid format or missing 'should_merge' key: {response_data}. Defaulting to NOT MERGE.")
                return False

        except Exception as e:
            logger.error(f"[LLM Merge Decision] An error occurred during LLM call: {e}. Defaulting to NOT MERGE.",
                         exc_info=True)
            return False
    def extract_entities_for_search(self, query: str) -> List[str]:
        """
        Extracts key entities from a natural language search query.
        This method uses a prompt optimized for simple keyword extraction.

        Args:
            query (str): The user's search query.

        Returns:
            List[str]: A list of extracted entity names.
        """
        logger.debug("--- LLM Call Start: Extracting Search Entities ---")

        search_results = self.llm.generate_response(
            messages=[
                {"role": "system", "content": SEARCH_ENTITY_EXTRACTION_PROMPT},
                {"role": "user", "content": query},
            ],
            tools=[SEARCH_ENTITY_EXTRACTION_TOOL],
        )

        logger.debug(f"LLM Response (Tool Calls):\n{json.dumps(search_results, indent=2, ensure_ascii=False)}")

        extracted_entities = []
        try:
            tool_calls = search_results.get("tool_calls", [])
            for tool_call in tool_calls:
                if tool_call.get("name") == "extract_search_entities":
                    arguments = tool_call.get("arguments", {})
                    entities = arguments.get("entities", [])
                    for entity_name in entities:
                        if entity_name:
                            clean_name = entity_name.lower().replace(" ", "_")
                            extracted_entities.append(clean_name)
        except Exception as e:
            logger.exception(f"Error parsing search entities: {e}")

        logger.debug(f"Extracted entity names for search: {extracted_entities}")
        return extracted_entities
