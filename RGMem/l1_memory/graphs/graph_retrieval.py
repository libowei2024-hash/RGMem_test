# file: l1_memory/graphs/graph_retrieval.py

import json
import logging
from typing import Any, Dict, List, Optional

from config import settings

try:
    from langchain_neo4j import Neo4jGraph
except ImportError:
    raise ImportError(
        "langchain_neo4j is not installed. Please install it using pip install langchain-neo4j"
    )

logger = logging.getLogger(__name__)


class GraphRetriever:
    """
    Retrieves and formats knowledge from a Neo4j graph, supporting complex queries and result formatting.
    """

    def __init__(
            self,
            graph_client: Neo4jGraph,
            llm_client: Any,
            embedding_model: Any,
            config: Any,
    ):
        self.graph = graph_client
        self.llm = llm_client
        self.embedding_model = embedding_model
        self.config = config
        self.retrieval_threshold = settings.retrieval_similar_threshold
        self.vector_search_limit = settings.vector_search_limit_per_entity

        if self.config.graph_store.config.base_label:
            self.node_label = ":`__Entity__`"
        else:
            self.node_label = ""

        logger.info("GraphRetriever initialized.")
        logger.info(f"  - Retrieval similarity threshold: {self.retrieval_threshold}")
        logger.info(f"  - Vector search limit per entity: {self.vector_search_limit}")

    def _build_cypher_query_parts(
            self, entity_queries: List[Dict], filters: Dict
    ) -> (str, Dict):
        """Builds the Cypher query and parameters for batch entity retrieval."""

        # Batch generate embeddings for all entity queries
        names_to_embed = [eq["name"] for eq in entity_queries]
        if hasattr(self.embedding_model, 'embed_batch'):
            logger.debug(f"Batch generating embeddings for {len(names_to_embed)} entities...")
            embeddings = self.embedding_model.embed_batch(names_to_embed)
        else:
            logger.warning("Embedding model does not support 'embed_batch', falling back to individual generation.")
            embeddings = [self.embedding_model.embed(name) for name in names_to_embed]

        unwind_params = []
        for i, eq in enumerate(entity_queries):
            unwind_params.append(
                {"query_name": eq["name"], "embedding": embeddings[i], "retrieve": eq["retrieve"], "index": i}
            )

        params = {
            "unwind_params": unwind_params,
            "threshold": self.retrieval_threshold,
            "limit_per_entity": self.vector_search_limit,
            "top_k_per_query": settings.retrieval_top_k,
            "user_id": filters.get("user_id"),
            "role_id": filters.get("role_id"),
        }

        agent_filter_str = "AND node.role_id = $role_id" if params["role_id"] else ""

        # Part 1: UNWIND and optimized Vector Search
        query = f"""
                       UNWIND $unwind_params AS query
                       CALL {{
                           WITH query
                           // Core optimization: Directly match nodes using the user_id property.
                           // This efficiently utilizes the B-Tree indexes on :event(user_id) and :abstract(user_id).
                           MATCH (node {{user_id: $user_id}})

                           // Perform fast label and property filtering on the small set of nodes returned by the index.
                           WHERE (node:event OR node:abstract) AND node.embedding IS NOT NULL {agent_filter_str}

                           // Now, the expensive vector computation is performed only on the small subset of nodes.
                           WITH node, query, vector.similarity.cosine(node.embedding, query.embedding) AS similarity
                           WHERE similarity >= $threshold
                           RETURN node, query AS query_out, similarity
                           ORDER BY similarity DESC
                           LIMIT $top_k_per_query
                       }}
                       WITH node, query_out AS query, similarity
                       """

        # Part 2: Pre-calculate combined summaries
        query += """
                        WITH node, query, similarity,
                             CASE 
                                 WHEN node:abstract AND node.summary IS NOT NULL 
                                 THEN [{name: node.name, summary: node.summary, last_update: node.last_summary_update}] 
                                 ELSE [] 
                             END AS combined_summaries
                        """

        # Part 3: Pre-calculate history and conclusions data
        query += """
        CALL {
            WITH node, query
            OPTIONAL MATCH (node)-[r:RELATED_TO]->(neighbor)
            WHERE ("history" IN query.retrieve OR "conclusions" IN query.retrieve OR "inference" IN query.retrieve)
              AND (neighbor.user_id = $user_id AND (neighbor.role_id = $role_id OR $role_id IS NULL))
            WITH node, query, collect({source: properties(node), rel: properties(r), target: properties(neighbor)}) as out_rels

            OPTIONAL MATCH (neighbor)-[r:RELATED_TO]->(node)
            WHERE ("history" IN query.retrieve OR "conclusions" IN query.retrieve OR "inference" IN query.retrieve)
              AND (neighbor.user_id = $user_id AND (neighbor.role_id = $role_id OR $role_id IS NULL))
            WITH out_rels, collect({source: properties(neighbor), rel: properties(r), target: properties(node)}) as in_rels

            RETURN [item IN (out_rels + in_rels) WHERE item.source IS NOT NULL AND item.target IS NOT NULL] AS all_rels
        }
        WITH node, query, similarity, combined_summaries, all_rels AS h_c_data_full
        """

        # Part 4: Final RETURN clause
        query += """
                RETURN query.query_name AS query_name, query.retrieve AS requested,
                       collect(DISTINCT {
                           node_properties: properties(node),
                           elementId: elementId(node),
                           similarity: similarity,
                           h_c_data: h_c_data_full,
                           summary_data: CASE WHEN "summary" IN query.retrieve THEN combined_summaries ELSE [] END
                       }) AS results
                """
        return query, params

    def _format_results(self, db_results: List[Dict]) -> Dict:
        """Merges raw Cypher output into a clean, entity-centric dictionary."""
        merged_entities = {}
        for row in db_results:
            query_name = row["query_name"]
            requested_fields = set(row["requested"])

            for res in row["results"]:
                node_id = res["elementId"]
                node_props = res["node_properties"]

                if node_id not in merged_entities:
                    merged_entities[node_id] = {
                        "node_properties": node_props,
                        "max_similarity": 0.0,
                        "matched_by_queries": set(),
                        "requested_fields": set(),
                        "summaries": {},
                        "inferences": {},
                        "one_hop_details": {}
                    }

                entity = merged_entities[node_id]
                entity["max_similarity"] = max(entity["max_similarity"], res["similarity"])
                entity["matched_by_queries"].add(query_name)
                entity["requested_fields"].update(requested_fields)

                if "summary" in requested_fields:
                    for summary_info in res.get("summary_data", []):
                        summary_key = summary_info['name']
                        if summary_key not in entity["summaries"]:
                            entity["summaries"][summary_key] = summary_info

                if "inference" in requested_fields:
                    for inference_info in res.get("h_c_data", []):
                        rel_props = inference_info.get("rel", {})
                        if rel_props.get("inference"):
                            inf_key = (inference_info["source"]["name"], inference_info["target"]["name"])
                            if inf_key not in entity["inferences"]:
                                entity["inferences"][inf_key] = {
                                    "source": inf_key[0], "target": inf_key[1],
                                    "inference": rel_props["inference"],
                                    "last_update": rel_props.get("last_inference_update")
                                }

                if "history" in requested_fields or "conclusions" in requested_fields:
                    for rel_detail in res.get("h_c_data", []):
                        rel_key = (
                        rel_detail["source"]["name"], rel_detail["rel"]["type"], rel_detail["target"]["name"])
                        if rel_key not in entity["one_hop_details"]:
                            entity["one_hop_details"][rel_key] = {
                                "source": rel_key[0], "relationship_type": rel_key[1], "target": rel_key[2],
                                "history": [], "conclusions": []
                            }
                        if "history" in requested_fields:
                            entity["one_hop_details"][rel_key]["history"].extend(
                                [json.loads(h) for h in rel_detail["rel"].get("history", [])])
                        if "conclusions" in requested_fields:
                            entity["one_hop_details"][rel_key]["conclusions"].extend(
                                [json.loads(c) for c in rel_detail["rel"].get("l1_conclusions", [])])

        final_output = {}
        for node_id, data in merged_entities.items():
            node_name = data["node_properties"]["name"]
            for rel_key, rel_data in data["one_hop_details"].items():
                rel_data["history"] = [dict(t) for t in {tuple(d.items()) for d in rel_data["history"]}]
                rel_data["conclusions"] = [dict(t) for t in {tuple(d.items()) for d in rel_data["conclusions"]}]

            final_output[node_name] = {
                "node_name": node_name,
                "similarity": data["max_similarity"],
                "matched_by": sorted(list(data["matched_by_queries"])),
                "summaries": list(data["summaries"].values()),
                "inferences": list(data["inferences"].values()),
                "one_hop_details": list(data["one_hop_details"].values())
            }
        return final_output

    def extract_knowledge(self, entity_queries: List[Dict], filters: Dict) -> Dict:
        """Main entry point for knowledge extraction, applying whitelist logic before querying."""
        user_id = filters.get("user_id")
        if not entity_queries or not user_id:
            logger.warning("Entity queries and a user_id are required for knowledge extraction.")
            return {}

        # Whitelist logic: if user is whitelisted, role_id is ignored for a global view
        effective_role_id = None
        if user_id not in settings.L1_USER_ID_WHITELIST:
            effective_role_id = filters.get("role_id")
        elif filters.get("role_id") is not None:
            logger.info(f"[Whitelist] User '{user_id}' is on the whitelist; retrieval will ignore role_id.")

        effective_filters = {"user_id": user_id, "role_id": effective_role_id}

        logger.info(f"Starting knowledge extraction for user '{user_id}' with {len(entity_queries)} entity queries.")
        query, params = self._build_cypher_query_parts(entity_queries, effective_filters)
        logger.debug(f"--- [Retriever] Generated Cypher Query ---\n{query}")

        try:
            db_results = self.graph.query(query, params=params)
            return self._format_results(db_results)
        except Exception as e:
            logger.error(f"An error occurred during graph retrieval: {e}", exc_info=True)
            return {"error": str(e)}

    def get_all(self, filters: Dict, limit: Optional[int] = None) -> List[Dict]:
        """Retrieves all nodes and relationships based on filters."""
        if limit is not None:
            logger.warning(
                f"Calling `get_all` with a `limit` of {limit} is ignored. "
                f"The global limit from settings ({settings.vector_search_limit_per_entity}) will be used."
            )

        agent_filter = ""
        params = {"user_id": filters["user_id"], "limit": settings.vector_search_limit_per_entity}
        if filters.get("role_id"):
            agent_filter = "AND n.role_id = $role_id AND m.role_id = $role_id"
            params["role_id"] = filters["role_id"]

        query = f"""
        MATCH (n {self.node_label} {{user_id: $user_id}})-[r]->(m {self.node_label} {{user_id: $user_id}})
        WHERE 1=1 {agent_filter}
        RETURN n.name AS source, type(r) AS relationship, m.name AS target
        LIMIT $limit
        """
        results = self.graph.query(query, params=params)
        final_results = [{"source": r["source"], "relationship": r["relationship"], "target": r["target"]} for r in
                         results]
        logger.info(f"Retrieved {len(final_results)} relationships")
        return final_results

    def format_results_as_text(self, results: dict, entity_queries: list) -> str:
        """Formats the structured, entity-centric results into a human-readable string."""
        output_lines = []
        if not results or results.get("error"):
            output_lines.append("\n❌ Retrieval failed or no results were found.")
            if results.get("error"):
                output_lines.append(f"   Error Message: {results['error']}")
            return "\n".join(output_lines)

        output_lines.append("\n✅ Retrieval successful! Here is the information gathered:")
        output_lines.append("=" * 60)
        seen_summaries = set()

        for entity_name, data in results.items():
            matched_by_str = ", ".join(data.get('matched_by', []))
            output_lines.append(
                f"\n▶️ Matched KG Entity: '{entity_name}' (found via '{matched_by_str}', max similarity: {data.get('similarity', 'N/A'):.4f})")
            output_lines.append("-" * 50)

            if data.get('summaries'):
                has_printed_summary_header = False
                for summary_info in data['summaries']:
                    summary_content = summary_info.get("summary", "")
                    if summary_content and summary_content not in seen_summaries:
                        if not has_printed_summary_header:
                            output_lines.append("\n    [L3 Summary]:")
                            has_printed_summary_header = True
                        output_lines.append(f"      - On the topic of '{summary_info['name']}':")
                        output_lines.append(f"        {summary_content}")
                        seen_summaries.add(summary_content)

            if data.get('inferences'):
                output_lines.append("\n    [L2 Inference]:")
                for inference_info in data['inferences']:
                    output_lines.append(
                        f"      - Regarding ('{inference_info['source']}') -> ('{inference_info['target']}'):")
                    output_lines.append(f"        {inference_info['inference']}")

            if data.get('one_hop_details'):
                has_content_to_render = any(d.get('history') or d.get('conclusions') for d in data['one_hop_details'])
                if has_content_to_render:
                    output_lines.append("\n    [L0/L1 Details - History & Conclusions]:")
                    for detail in data['one_hop_details']:
                        if not detail.get('history') and not detail.get('conclusions'):
                            continue
                        rel_content_display = "N/A"
                        if detail.get('history') and detail['history'][0].get('relationship_content'):
                            rel_content_display = detail['history'][0]['relationship_content']
                        output_lines.append(
                            f"      - Relationship: ('{detail['source']}') -[{detail['relationship_type']}: {rel_content_display}]-> ('{detail['target']}')")

                        if detail.get('history'):
                            history_list = detail['history']
                            if detail.get('relationship_type') == 'classification':
                                history_to_print = sorted(history_list, key=lambda x: x.get('timestamp', ''),
                                                          reverse=True)[:1]
                                output_lines.append(f"        - Mention History (1 time, type: classification):")
                            else:
                                history_to_print = history_list
                                output_lines.append(f"        - Mention History ({len(history_list)} times):")
                            for hist_item in history_to_print:
                                output_lines.append(
                                    f"          - Content: '{hist_item.get('relationship_content', 'N/A')}', Timestamp: {hist_item.get('timestamp', 'N/A')}")

                        if detail.get('conclusions'):
                            output_lines.append("        - High-value Conclusions:")
                            for conc in detail['conclusions']:
                                output_lines.append(
                                    f"          - {conc['content']} (from memory: {conc['source_memory_id']})")
        output_lines.append("\n" + "=" * 60)
        return "\n".join(output_lines)