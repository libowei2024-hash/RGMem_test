# l1_memory/data_loader.py

from elasticsearch import Elasticsearch
from config import settings
from l1_memory.utils import logger
import urllib3


class L0MemoryLoader:
    """Loads and transforms L0 memory data from Elasticsearch."""

    def __init__(self):
        try:
            # Reuse ES configuration from settings
            self.es_client = Elasticsearch(
                hosts=[settings.ES_CONFIG["host"]],
                basic_auth=(settings.ES_CONFIG["user"], settings.ES_CONFIG["password"]),
                request_timeout=30,
                verify_certs=False,
            )

            # Disable SSL warnings
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

            self.index_name = settings.ES_INDEX_TOPICAL_L0
            if not self.es_client.ping():
                raise ConnectionError("Unable to connect to Elasticsearch instance.")
            logger.info(
                f"L0MemoryLoader successfully connected to Elasticsearch, reading from index '{self.index_name}'.")
        except Exception as e:
            logger.error(f"Failed to initialize L0MemoryLoader: {e}", exc_info=True)
            raise

    def get_memory_by_id(self, memory_id: str) -> dict | None:
        """
        Retrieves an L0 memory from Elasticsearch by memory_id and converts it to the format required by L1.
        """
        try:
            response = self.es_client.get(index=self.index_name, id=memory_id)
            source = response.get('_source')
            if not source:
                logger.warning(f"Found memory_id '{memory_id}' in ES, but the document content is empty.")
                return None

            # Transform data format
            l1_formatted_data = {
                "memory_id": source.get("memory_id"),
                "user_id": source.get("metadata", {}).get("user_id"),
                "role_id": source.get("metadata", {}).get("role_id"),
                "created_at": source.get("metadata", {}).get("created_at"),
                "core_fact": source.get("core_fact"),
                "conclusions": source.get("conclusions", [])
            }

            # Validate critical fields
            if not all([l1_formatted_data["memory_id"], l1_formatted_data["user_id"], l1_formatted_data["core_fact"]]):
                logger.error(
                    f"Memory '{memory_id}' retrieved from ES is missing critical fields (memory_id, user_id, core_fact).")
                return None

            logger.info(f"Successfully retrieved and transformed data for memory_id '{memory_id}' from ES.")
            return l1_formatted_data

        except Exception as e:
            logger.error(f"Failed to retrieve data from ES for memory_id '{memory_id}': {e}", exc_info=True)
            return None