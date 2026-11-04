# file: l2_retrieval/l1_adapter.py

from typing import List, Dict, Optional
from config import settings
from l1_memory.memory.main import Memory
from l1_memory.configs.base import MemoryConfig
from l1_memory.utils import logger

class L1Adapter:
    """
    An adapter to provide a standardized interface for retrieving knowledge
    from the L1 memory layer (MemoryGraph).
    """
    def __init__(self, l1_memory: Memory):
        """Initializes the L1 Memory instance."""
        try:
            self.l1_memory = l1_memory
            logger.info("[L2-L1 Adapter] L1 Memory instance initialized successfully.")
        except Exception as e:
            logger.critical("[L2-L1 Adapter] Failed to initialize L1 Memory.", exc_info=True)
            self.l1_memory = None

    def retrieve(self, user_id: str, role_id: int, entity_queries: List[Dict]) -> Optional[Dict]:
        """
        Retrieves knowledge from the L1 Knowledge Graph and returns the raw structured result dictionary.
        """
        if not self.l1_memory:
            logger.warning("L1 Retrieval Skipped: L1 Memory failed to initialize.")
            return None

        logger.info(f"[L2-L1 Adapter] Retrieving L1 knowledge for user '{user_id}' with {len(entity_queries)} queries.")
        try:
            # Directly return the raw dictionary, skipping formatting
            raw_results = self.l1_memory.retrieve(
                entity_queries=entity_queries,
                user_id=user_id,
                role_id=role_id
            )
            return raw_results
        except Exception as e:
            logger.error(f"[L2-L1 Adapter] An error occurred during L1 retrieval.", exc_info=True)
            return None