# file: l2_retrieval/l0_adapter.py

from typing import List, Dict, Optional
from l0_memory.rag_system.rag_server import RagServer
from l0_memory.utils.logger import file_logger as logger


class L0Adapter:
    """
    An adapter providing a standardized interface for retrieving context
    from the L0 memory layer (RagServer), supporting both relevance and recency.
    """

    def __init__(self):
        """Initializes the L0 RagServer instance."""
        try:
            self.rag_server = RagServer()
            logger.info("[L2-L0 Adapter] L0 RagServer instance initialized successfully.")
        except Exception as e:
            logger.critical("[L2-L0 Adapter] Failed to initialize L0 RagServer.", exc_info=True)
            self.rag_server = None

    def retrieve_relevance(self, user_id: str, role_id: int, query: str, keywords: str) -> Optional[
        Dict[str, List[Dict]]]:
        """
        Retrieves relevance-based context from L0 and returns raw structured data dictionary.

        Return format: {"core_facts": [...], "conclusions": [...]}
        """
        if not self.rag_server:
            logger.warning("L0 Relevance Retrieval Skipped: RagServer failed to initialize.")
            return None

        logger.info(f"[L2-L0 Adapter] Retrieving L0 RELEVANCE context for user '{user_id}'")
        try:
            # Directly call and return the raw dictionary result from get_relevance_context
            return self.rag_server.get_relevance_context(
                user_id=user_id,
                role_id=role_id,
                query=query,
                keywords=keywords
            )
        except Exception as e:
            logger.error(f"[L2-L0 Adapter] An error occurred during L0 relevance retrieval.", exc_info=True)
            return None

    def retrieve_recency(self, user_id: str, role_id: int) -> Optional[List[Dict]]:
        """
        Retrieves recency-based context (latest core facts) from L0 and returns raw document list.

        Return format: [...]
        """
        if not self.rag_server:
            logger.warning("L0 Recency Retrieval Skipped: RagServer failed to initialize.")
            return None

        logger.info(f"[L2-L0 Adapter] Retrieving L0 RECENCY context for user '{user_id}'.")
        try:
            # Directly call and return the raw list result from get_recency_context
            return self.rag_server.get_recency_context(user_id=user_id, role_id=role_id)
        except Exception as e:
            logger.error(f"[L2-L0 Adapter] An error occurred during L0 recency retrieval.", exc_info=True)
            return None