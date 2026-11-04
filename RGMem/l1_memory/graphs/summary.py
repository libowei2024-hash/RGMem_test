# l1_memory/graphs/summary.py

import logging
import json
import time
from typing import Optional

from l1_memory.graphs.utils import UPDATE_COMMUNITY_SUMMARY_PROMPT, UPDATE_HIERARCHICAL_SUMMARY_PROMPT

logger = logging.getLogger(__name__)


class SummaryProcessor:
    def __init__(self, graph_client, llm_client):
        self.graph = graph_client
        self.llm = llm_client
        logger.info("SummaryProcessor initialized.")

    def update_community_summary(self, node_id: str, trigger_timestamp: str):
        logger.info(f"--- [Sync Summary Task STARTED] for abstract node: {node_id} ---")
        try:
            node_query = "MATCH (n:abstract) WHERE elementId(n) = $node_id RETURN n.name AS name, n.summary AS old_summary, n.last_summary_update AS last_update"
            node_data = self.graph.query(node_query, params={"node_id": node_id})
            if not node_data:
                logger.warning(f"[Sync Summary Task FAILED] Node {node_id} not found. Exiting.")
                return

            node_name = node_data[0]['name']
            old_summary = node_data[0].get('old_summary') or ""
            last_update_ts = node_data[0].get('last_update') or "1970-01-01T00:00:00Z"

            logger.info(f"[Sync Summary] Context for {node_name} ({node_id}):")
            logger.info(f"  - Old Summary: '{old_summary[:50]}...'")
            logger.info(f"  - Last Update Timestamp: {last_update_ts}")

            relations_query = """
            MATCH (n:abstract)-[r:RELATED_TO]->(e:event) WHERE elementId(n) = $node_id
            RETURN r.inference AS inference, r.last_inference_update AS last_inf_update,
                   r.l1_conclusions AS conclusions, r.history AS history,
                   e.name AS event_name
            """
            relations_data = self.graph.query(relations_query, params={"node_id": node_id})
            logger.info(f"[Sync Summary] Found {len(relations_data)} related event relationships for node {node_name}.")

            new_evidence_collection = []
            latest_evidence_timestamp = last_update_ts

            for rel in relations_data:
                event_name = rel.get('event_name')
                if rel.get('inference') and (rel.get('last_inf_update') or '1970-01-01T00:00:00Z') > last_update_ts:
                    new_evidence_collection.append(f"- [关于 '{event_name}' 的推断]: {rel['inference']}")
                    if rel['last_inf_update'] > latest_evidence_timestamp:
                        latest_evidence_timestamp = rel['last_inf_update']
                    continue

                new_conclusions = [json.loads(c) for c in rel.get('conclusions', []) if
                                   json.loads(c)['timestamp'] > last_update_ts]
                if new_conclusions:
                    for c in sorted(new_conclusions, key=lambda x: x['timestamp'], reverse=True)[:2]:
                        new_evidence_collection.append(f"- [关于 '{event_name}' 的结论]: {c['content']}")
                        if c['timestamp'] > latest_evidence_timestamp:
                            latest_evidence_timestamp = c['timestamp']
                    continue

                new_history = [json.loads(h) for h in rel.get('history', []) if
                               json.loads(h)['timestamp'] > last_update_ts]
                if new_history:
                    for h in sorted(new_history, key=lambda x: x['timestamp'], reverse=True)[:2]:
                        new_evidence_collection.append(
                            f"- [关于 '{event_name}' 的原始事件]: 关系是 '{h['relationship_content']}'")
                        if h['timestamp'] > latest_evidence_timestamp:
                            latest_evidence_timestamp = h['timestamp']

            if not new_evidence_collection:
                logger.info(
                    f"[Sync Summary Task SKIPPED] No new evidence found for community summary of {node_name}. Exiting.")
                if trigger_timestamp > last_update_ts:
                    ts_update_query = "MATCH (n) WHERE elementId(n) = $id SET n.last_summary_update = $ts"
                    self.graph.query(ts_update_query, params={"id": node_id, "ts": trigger_timestamp})
                return

            new_evidence_str = "\n".join(new_evidence_collection)
            prompt = UPDATE_COMMUNITY_SUMMARY_PROMPT.format(
                core_entity_name=node_name,
                old_summary=old_summary,
                new_evidence_str=new_evidence_str
            )

            response = self.llm.generate_response(messages=[{"role": "user", "content": prompt}])
            new_summary = response if isinstance(response, str) else response.get("content", "")

            if not new_summary:
                logger.warning(f"[Sync Summary Task FAILED] LLM returned invalid content for {node_name}. Exiting.")
                return

            update_query = "MATCH (n:abstract) WHERE elementId(n) = $node_id SET n.summary = $new_summary, n.last_summary_update = $new_ts"
            self.graph.query(update_query, params={"node_id": node_id, "new_summary": new_summary, "new_ts": latest_evidence_timestamp})
            logger.info(f"Successfully updated community summary for node {node_name} ({node_id})")

            self.propagate_dirty_flag_to_direct_parents(node_id)
            logger.info(f"--- [Sync Summary Task COMPLETED] for node: {node_id} ---")

        except Exception as e:
            logger.critical(f"--- [Sync Summary Task CRASHED] for {node_id}: {e} ---", exc_info=True)

    def propagate_dirty_flag_to_direct_parents(self, node_id: str):
        try:
            query = """
            MATCH (p:abstract)-[:RELATED_TO]->(c:abstract)
            WHERE elementId(c) = $node_id AND p <> c AND NOT p.is_dirty
            SET p.is_dirty = true
            """
            self.graph.query(query, params={"node_id": node_id})
            logger.info(f"Propagated dirty flag upwards from node {node_id} to its direct parents.")
        except Exception as e:
            logger.error(f"Failed to propagate dirty flag from node {node_id}: {e}", exc_info=True)

    def summarize_dirty_nodes(self, user_id: str, role_id: Optional[str], trigger_timestamp: str):
        log_context = f"user '{user_id}'"
        if role_id:
            log_context += f" and agent '{role_id}'"

        logger.info(f"Starting hierarchical summary update process for {log_context}.")

        processing_label = f"Processing_{int(time.time() * 1000)}"

        try:
            pass_count = 0
            MAX_PASSES = 20

            while pass_count < MAX_PASSES:
                pass_count += 1
                logger.info(f"--- Starting Hierarchical Summary Pass #{pass_count} for {log_context} ---")

                agent_filter_str = "AND n.role_id = $role_id" if role_id else ""

                query = f"""
                MATCH (n:abstract {{is_dirty: true, user_id: $user_id}})
                WHERE NOT n:{processing_label} {agent_filter_str}
                OPTIONAL MATCH (n)-[:RELATED_TO]->(child:abstract {{is_dirty: true, user_id: $user_id}})
                WHERE NOT child:{processing_label} AND (child.role_id = $role_id OR $role_id IS NULL)
                WITH n, count(child) AS dirty_children_count
                WHERE dirty_children_count = 0
                RETURN elementId(n) AS node_id
                LIMIT 100
                """
                params = {"user_id": user_id, "role_id": role_id}
                nodes_to_process_ids = [res['node_id'] for res in self.graph.query(query, params=params)]

                if not nodes_to_process_ids:
                    logger.info("No more nodes are ready for update in this pass. Process is stable. Exiting loop.")
                    break

                logger.info(f"Found {len(nodes_to_process_ids)} nodes to process in this pass.")

                self.graph.query(
                    f"MATCH (n) WHERE elementId(n) IN $ids SET n:{processing_label}",
                    params={"ids": nodes_to_process_ids}
                )

                for node_id in nodes_to_process_ids:
                    try:
                        data_query = """
                        MATCH (n:abstract) WHERE elementId(n) = $node_id
                        OPTIONAL MATCH (n)-[:RELATED_TO]->(child:abstract)
                        WHERE child.user_id = $user_id AND (child.role_id = $role_id OR $role_id IS NULL) AND child.summary IS NOT NULL
                        RETURN n.name AS name, n.summary AS old_summary,
                               collect({
                                   name: child.name, 
                                   summary: child.summary, 
                                   last_update: child.last_summary_update
                               }) AS child_details
                        """
                        node_data_result = self.graph.query(data_query, params={"node_id": node_id, "user_id": user_id, "role_id": role_id})


                        if not node_data_result: continue

                        node = node_data_result[0]
                        child_details = [cd for cd in node.get('child_details', []) if cd.get('name')]

                        if not child_details:
                            logger.warning(f"Node '{node['name']}' is dirty but has no children with summaries. Skipping summary generation.")
                            continue

                        latest_child_update_ts = max([child.get('last_update') or "1970-01-01T00:00:00Z" for child in child_details])
                        child_summaries_str = "\n".join([f"- 子概念 '{child['name']}': {child['summary']}" for child in child_details])
                        prompt = UPDATE_HIERARCHICAL_SUMMARY_PROMPT.format(
                            core_entity_name=node['name'],
                            old_summary=node.get('old_summary') or "",
                            child_summaries_str=child_summaries_str
                        )
                        response = self.llm.generate_response(messages=[{"role": "user", "content": prompt}])
                        new_summary = response if isinstance(response, str) else response.get("content", "")

                        if not new_summary:
                            logger.warning(f"LLM failed to generate hierarchical summary for '{node['name']}'. Skipping.")
                            continue

                        update_query = f"""
                        MATCH (n:abstract) WHERE elementId(n) = $node_id
                        SET n.summary = $new_summary,
                            n.last_summary_update = $ts,
                            n.is_dirty = false
                        WITH n
                        MATCH (p:abstract)-[:RELATED_TO]->(n)
                        WHERE p <> n AND NOT p:{processing_label} AND NOT p.is_dirty
                        SET p.is_dirty = true
                        """
                        self.graph.query(update_query, params={
                            "node_id": node_id,
                            "new_summary": new_summary,
                            "ts": latest_child_update_ts
                        })
                        logger.info(f"Successfully updated summary for '{node['name']}' and propagated dirty flag.")

                    except Exception as e:
                        logger.error(f"Failed to process node {node_id} during hierarchical summary update.", exc_info=True)

        finally:
            cleanup_query = f"MATCH (n:{processing_label}) WHERE n.user_id = $user_id REMOVE n:{processing_label}"
            self.graph.query(cleanup_query, params={"user_id": user_id})
            logger.info(f"Cleaned up all :{processing_label} labels for {log_context}.")

        if pass_count >= MAX_PASSES:
            logger.error(f"Hierarchical summary update for {log_context} reached max passes ({MAX_PASSES}). Terminating to prevent potential infinite loop.")
        else:
            logger.info(f"Hierarchical summary update process for {log_context} finished after {pass_count} passes.")