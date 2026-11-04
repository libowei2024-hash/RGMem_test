# l0_memory/history_db/elasticsearch.py

from typing import List, Dict, Optional, Any
from l0_memory.history_db.base import BaseHistoryDB
from l0_memory.utils.elasticsearch_client import ElasticsearchClient
from config import settings
from l0_memory.utils.logger import file_logger as logger


class ElasticsearchHistoryDB(BaseHistoryDB):
    """
    Elasticsearch implementation of the historical dialogue database.
    Query logic is based on user_id and role_id.
    """

    def __init__(self):
        self.es_client = ElasticsearchClient().es
        self.index_name = settings.ES_INDEX_CHAT_HISTORY
        logger.info(f"[ElasticsearchHistoryDB] Initialized, using index: {self.index_name}")

    def _build_query(self, user_id: str, role_id: int, start_ts: Optional[str], end_ts: Optional[str]) -> dict:
        """Builds a query based on user_id, role_id, and time range."""
        filters = [
            {"term": {"user_id": user_id}},
            {"term": {"role_id": role_id}},
            {"terms": {"role": ["user", "assistant"]}}
        ]
        time_range = {}
        if start_ts:
            time_range["gt"] = start_ts
        if end_ts:
            time_range["lte"] = end_ts

        if time_range:
            filters.append({"range": {"created_at": time_range}})

        return {"bool": {"filter": filters}}

    def get_dialogues_between(
            self,
            user_id: str,
            role_id: int,
            start_marker_created_at: Optional[str],
            end_marker_created_at: Optional[str],
            limit: int
    ) -> List[Dict[str, Any]]:
        """Retrieves complete dialogue records within a specified time range."""
        query = self._build_query(user_id, role_id, start_marker_created_at, end_marker_created_at)
        session_key = f"{user_id}_{role_id}"

        try:
            response = self.es_client.search(
                index=self.index_name,
                query=query,
                sort=[{"created_at": "asc"}],
                size=limit
            )
            hits = response.get('hits', {}).get('hits', [])
            results = []
            for doc in hits:
                doc_data = doc['_source']
                doc_data['message_id'] = doc['_id']
                results.append(doc_data)
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve dialogues from ES (session_key: {session_key}): {e}", exc_info=True)
            return []

    def get_dialogue_count_between(
            self,
            user_id: str,
            role_id: int,
            start_marker_created_at: Optional[str],
            end_marker_created_at: Optional[str]
    ) -> int:
        """Retrieves the count of dialogue records within a specified time range."""
        query = self._build_query(user_id, role_id, start_marker_created_at, end_marker_created_at)
        session_key = f"{user_id}_{role_id}"

        try:
            response = self.es_client.count(
                index=self.index_name,
                query=query
            )
            return response.get('count', 0)
        except Exception as e:
            logger.error(f"Failed to retrieve dialogue count from ES (session_key: {session_key}): {e}", exc_info=True)
            return 0