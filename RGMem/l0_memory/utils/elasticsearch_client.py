# l0_memory/utils/elasticsearch_client.py

import os
from elasticsearch import Elasticsearch
import urllib3
from config import settings
from .logger import file_logger as logger


class ElasticsearchClient:
    """
    Elasticsearch client wrapper providing environment sanitization and helper methods
    for dialogue history and memory indexing.
    """

    def __init__(self):
        try:
            # Clear proxy environment variables to ensure direct connection to Elasticsearch.
            # This handles cases where local connections are blocked by system-wide proxies.
            proxy_keys = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']
            for key in proxy_keys:
                if key in os.environ:
                    logger.info(f"Cleared proxy environment variable: {key}")
                    del os.environ[key]

            self.es = Elasticsearch(
                hosts=[settings.ES_CONFIG["host"]],
                basic_auth=(settings.ES_CONFIG["user"], settings.ES_CONFIG["password"]),
                request_timeout=30,
                verify_certs=False,
                ssl_show_warn=False
            )

            # Disable InsecureRequestWarning caused by verify_certs=False
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

            if not self.es.ping():
                raise ConnectionError("Could not connect to Elasticsearch instance.")

            logger.info(f"Successfully connected to Elasticsearch: {settings.ES_CONFIG['host']}")

        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch client: {e}", exc_info=True)
            raise

    def get_latest_turn_and_history(self, num_history_items: int) -> tuple[int, list, str]:
        """
        Retrieves the latest turn number, formatted dialogue history, and last user message
        from the permanent chat log index.
        """
        index_name = settings.ES_INDEX_CHAT_HISTORY
        if not self.es.indices.exists(index=index_name):
            logger.warning(f"Index '{index_name}' does not exist. Starting from scratch.")
            return 0, [], "Hello! Let's chat."

        try:
            resp = self.es.search(
                index=index_name,
                size=num_history_items,
                sort=[{"turn": {"order": "desc"}}]
            )

            hits = resp['hits']['hits']
            if not hits:
                logger.info(f"Index '{index_name}' is empty. Starting from scratch.")
                return 0, [], "Hello! Let's chat."

            latest_turn = hits[0]['_source']['turn']

            # Determine the last relevant user message for context continuation
            last_user_message = "Let's continue our conversation."
            if len(hits) > 1 and hits[1]['_source']['role'] == 'user':
                last_user_message = hits[1]['_source']['message']
            elif hits[0]['_source']['role'] == 'user':
                last_user_message = hits[0]['_source']['message']

            # Rebuild chat history for LLM prompt context in correct chronological order
            chat_history_for_llms = []
            for hit in reversed(hits):
                role = hit['_source']['role']
                message = hit['_source']['message']
                formatted_role = "User" if role == "user" else "Assistant"
                chat_history_for_llms.append(f"{formatted_role}: {message}")

            logger.info(f"History loaded from ES. Latest turn: {latest_turn}, Items loaded: {len(chat_history_for_llms)}")
            return latest_turn, chat_history_for_llms, last_user_message

        except Exception as e:
            logger.error(f"Failed to load history from ES: {e}", exc_info=True)
            return 0, [], "Hello! Let's chat."

    def index_document(self, index_name: str, document: dict, document_id: str = None):
        """Indexes a document into the specified Elasticsearch index."""
        try:
            self.es.index(index=index_name, id=document_id, document=document)
        except Exception as e:
            logger.error(f"Failed to write to index '{index_name}': {e}", exc_info=True)

    def get_latest_turn_for_session(self, session_id: str) -> int:
        """Retrieves the latest turn number for a specific session ID."""
        try:
            resp = self.es.search(
                index=settings.ES_INDEX_CHAT_HISTORY,
                size=1,
                query={
                    "term": {
                        "session_id.keyword": session_id
                    }
                },
                sort=[{"turn": {"order": "desc"}}]
            )
            hits = resp.get('hits', {}).get('hits', [])
            if hits:
                latest_turn = hits[0]['_source'].get('turn', 0)
                logger.info(f"Recovered turn {latest_turn} for session '{session_id}'")
                return latest_turn
            return 0
        except Exception as e:
            logger.warning(f"Failed to query latest turn for session '{session_id}', starting from 0: {e}")
            return 0

    def index_l0_memory(self, memory_doc: dict):
        """Indexes a formatted L0 memory document into Elasticsearch."""
        index_name = settings.ES_INDEX_TOPICAL_L0
        doc_id = memory_doc.get("memory_id")
        try:
            self.es.index(index=index_name, id=doc_id, document=memory_doc)
            logger.info(f"Successfully indexed L0 memory to '{index_name}' (ID: {doc_id})")
        except Exception as e:
            logger.error(f"Failed to write L0 memory to index '{index_name}': {e}", exc_info=True)