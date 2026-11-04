# file: l1_memory/graphs/inference.py

import logging
import json  # 确保导入 json
from l1_memory.graphs.utils import UPDATE_INFERENCE_PROMPT

logger = logging.getLogger(__name__)


class InferenceProcessor:
    def __init__(self, graph_client, llm_client):
        self.graph = graph_client
        self.llm = llm_client
        logger.info("InferenceProcessor initialized.")

    def update_inference(self, relationship_id: str, trigger_timestamp: str):
        logger.info(f"--- [Sync Inference Task STARTED] for relationship: {relationship_id} ---")
        try:
            query = """
            MATCH (s)-[r]->(d) WHERE elementId(r) = $rel_id
            RETURN
                s.name AS source_name,
                d.name AS dest_name,
                r.inference AS old_inference,
                r.last_inference_update AS last_update,
                r.history AS history,
                r.l1_conclusions AS conclusions
            """
            result = self.graph.query(query, params={"rel_id": relationship_id})
            if not result:
                logger.warning(f"[Sync Inference Task FAILED] Relationship {relationship_id} not found. Exiting.")
                return

            data = result[0]
            source_name = data.get('source_name')
            dest_name = data.get('dest_name')
            old_inference = data.get('old_inference') or ""
            last_update_ts = data.get('last_update') or "1970-01-01T00:00:00Z"

            logger.info(
                f"[Sync Inference] Context for relationship between '{source_name}' and '{dest_name}' ({relationship_id}):")
            logger.info(f"  - Old Inference: '{old_inference[:50]}...'")
            logger.info(f"  - Last Update Timestamp: {last_update_ts}")
            logger.info(f"  - Total history items: {len(data.get('history', []))}")
            logger.info(f"  - Total conclusion items: {len(data.get('conclusions', []))}")

            new_evidence_items = []
            latest_entry_timestamp = last_update_ts

            for h_str in data.get('history', []):
                h = json.loads(h_str)
                if h['timestamp'] > last_update_ts:
                    new_evidence_items.append(f"- [Raw Event]: {h['relationship_content']} (at {h['timestamp']})")
                    if h['timestamp'] > latest_entry_timestamp:
                        latest_entry_timestamp = h['timestamp']

            for c_str in data.get('conclusions', []):
                c = json.loads(c_str)
                if c['timestamp'] > last_update_ts:
                    new_evidence_items.append(f"- [Conclusion]: {c['content']} (at {c['timestamp']})")
                    if c['timestamp'] > latest_entry_timestamp:
                        latest_entry_timestamp = c['timestamp']

            if not new_evidence_items:
                logger.info(
                    f"[Sync Inference Task SKIPPED] No new evidence found for {relationship_id} since {last_update_ts}. Exiting.")
                if trigger_timestamp > last_update_ts:
                    update_ts_query = "MATCH ()-[r]->() WHERE elementId(r) = $rel_id SET r.last_inference_update = $ts"
                    self.graph.query(update_ts_query, params={"rel_id": relationship_id, "ts": trigger_timestamp})
                return

            logger.info(
                f"[Sync Inference] Found {len(new_evidence_items)} new pieces of evidence for {relationship_id}.")

            new_evidence_str = "\n".join(new_evidence_items)

            # [核心修改] 在格式化Prompt时，传入实体名称
            prompt = UPDATE_INFERENCE_PROMPT.format(
                source_entity_name=source_name,
                destination_entity_name=dest_name,
                old_inference=old_inference,
                new_evidence=new_evidence_str
            )

            logger.info(
                f"[Sync Inference] Calling LLM to generate new inference for relationship between '{source_name}' and '{dest_name}'...")
            response = self.llm.generate_response(messages=[{"role": "user", "content": prompt}])
            new_inference = response if isinstance(response, str) else response.get("content", "")

            if not new_inference or "no new inference" in new_inference.lower():
                logger.warning(
                    f"[Sync Inference Task FAILED] LLM returned invalid content for {relationship_id}. Exiting.")
                return

            logger.info(f"[Sync Inference] LLM generated new inference: '{new_inference[:100]}...'")

            update_query = "MATCH ()-[r]->() WHERE elementId(r) = $rel_id SET r.inference = $new_inference, r.last_inference_update = $new_ts"
            self.graph.query(update_query, params={"rel_id": relationship_id, "new_inference": new_inference,
                                                   "new_ts": latest_entry_timestamp})
            logger.info(f"--- [Sync Inference Task COMPLETED] Successfully updated inference for {relationship_id} ---")

        except Exception as e:
            logger.critical(f"--- [Sync Inference Task CRASHED] for {relationship_id}: {e} ---", exc_info=True)