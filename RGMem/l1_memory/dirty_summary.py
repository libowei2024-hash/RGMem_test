# file: dirty_summary.py

import os
from datetime import datetime, timezone
from l1_memory.configs.base import MemoryConfig
from l1_memory.graphs.neo4j import MemoryGraph
from l1_memory.utils import logger
from config import settings

class DirtySummaryScheduler:
    """
    A service class responsible for executing global, hierarchical summary update tasks.
    """
    def __init__(self, memory_graph: MemoryGraph):
        logger.info("Initializing DirtySummaryScheduler...")
        try:
            # Build MemoryConfig object from settings
            config = MemoryConfig(
                llm=settings.L1_LLM_CONFIG,
                embedder=settings.L1_EMBEDDER_CONFIG,
                graph_store=settings.L1_GRAPH_STORE_CONFIG
            )
            # Initialize MemoryGraph instance
            self.graph_instance = memory_graph
            self.summary_processor = self.graph_instance.summary_processor
            logger.info("DirtySummaryScheduler initialized successfully.")
        except Exception as e:
            logger.critical("DirtySummaryScheduler initialization failed.", exc_info=True)
            raise

    def run_batch_processing(self):
        """
        Main execution method of the scheduler.
        """
        logger.info("--- [Global Hierarchical Summary Task] Task Started ---")
        try:
            logger.info("Searching Knowledge Graph for all user/agent combinations containing 'dirty' nodes...")
            query = """
            MATCH (n:abstract {is_dirty: true})
            RETURN DISTINCT n.user_id AS user_id, n.role_id AS role_id
            """
            id_combinations = self.graph_instance.graph.query(query)

            if not id_combinations:
                logger.info("No 'dirty' nodes requiring hierarchical summary updates found in Knowledge Graph. Task ended.")
                return

            logger.info(f"Found {len(id_combinations)} user/agent combinations requiring hierarchical summary updates.")

            trigger_timestamp = datetime.now(timezone.utc).isoformat()
            for combo in id_combinations:
                user_id = combo['user_id']
                role_id = combo['role_id']

                log_context = f"user '{user_id}'"
                if role_id:
                    log_context += f" and agent '{role_id}'"

                logger.info(f"\n--- Processing combination: {log_context} ---")
                self.summary_processor.summarize_dirty_nodes(
                    user_id=user_id,
                    role_id=role_id,
                    trigger_timestamp=trigger_timestamp
                )
                logger.info(f"--- Combination processed: {log_context} ---")

            logger.info("\n--- [Global Hierarchical Summary Task] All combinations processed, task successfully completed ---")

        except Exception as e:
            logger.critical(f"--- [Global Hierarchical Summary Task] Critical error occurred, task aborted ---", exc_info=True)