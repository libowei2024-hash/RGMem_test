# scripts/trigger_dirty_summary.py

import os
import sys

# --- Setup Python Path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# --- Core Module Imports ---
from config import settings
from l1_memory.dirty_summary import DirtySummaryScheduler
from l1_memory.graphs.neo4j import MemoryGraph
from l1_memory.configs.base import MemoryConfig
from l0_memory.utils.logger import console_logger


def main():
    """
    Script to manually trigger the L1 hierarchical summary update process.
    """
    console_logger.info("--- Manual Trigger for L1 Hierarchical Summary Update Started ---")

    # Step 1: Initialize L1 core components
    # This logic matches the initialization in main.py lifespan or other scripts
    # to ensure we are operating on the same knowledge graph configuration.
    console_logger.info("1. Initializing L1 MemoryGraph core components...")
    try:
        # Create unified L1 configuration object
        l1_config = MemoryConfig(
            llm=settings.L1_LLM_CONFIG,
            embedder=settings.L1_EMBEDDER_CONFIG,
            graph_store=settings.L1_GRAPH_STORE_CONFIG
        )
        # Create MemoryGraph instance
        memory_graph_instance = MemoryGraph(config=l1_config)
        console_logger.info("   - MemoryGraph initialized successfully.")

        # Create DirtySummaryScheduler instance injected with MemoryGraph
        scheduler_instance = DirtySummaryScheduler(memory_graph=memory_graph_instance)
        console_logger.info("   - DirtySummaryScheduler initialized successfully.")

    except Exception as e:
        console_logger.critical(f"Failed to initialize L1 components, script cannot proceed: {e}", exc_info=True)
        return

    # Step 2: Execute summary update process
    console_logger.info("\n2. Preparing to execute 'run_batch_processing'...")
    console_logger.info("   - This process scans the graph for all nodes marked 'is_dirty: true'.")
    console_logger.info("   - It updates hierarchical summaries iteratively from the bottom up until all summaries are current.")

    try:
        # Execute the core method
        scheduler_instance.run_batch_processing()

        console_logger.info("\n--- Manual hierarchical summary update task completed successfully. ---")

    except Exception as e:
        console_logger.error(f"Error occurred during 'run_batch_processing': {e}", exc_info=True)
        console_logger.info("\n--- Hierarchical summary update task aborted due to error. ---")


if __name__ == "__main__":
    main()