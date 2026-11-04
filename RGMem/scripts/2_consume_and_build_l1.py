# scripts/2_consume_and_build_l1.py

import os
import sys
from tqdm import tqdm

# --- Setup Python Path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# --- Core Module Imports ---
from config import settings
from l1_memory.data_loader import L0MemoryLoader
from l1_memory.configs.base import MemoryConfig
from l1_memory.graphs.neo4j import MemoryGraph
from l0_memory.utils.logger import console_logger

# ==============================================================================
#  >>>>> Configuration <<<<<
# ==============================================================================
# The directory containing the 'l0_memory_ids.txt' file.
# This should match the directory containing LOCAL_HISTORY_FILE_PATH in script 1.
DATA_WORKING_DIRECTORY = ""
MEMORY_ID_FILENAME = "l0_memory_ids.txt"

# ==============================================================================
#  >>>>> Core Implementation <<<<<
# ==============================================================================

def load_memory_ids_from_file(filepath: str) -> list:
    """Loads all memory_ids from a local text file, preserving their original order."""
    if not os.path.exists(filepath):
        console_logger.error(f"Error: Memory ID file '{filepath}' not found.")
        console_logger.error("Please run '1_produce_l0_memories.py' first to generate this file.")
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Read lines and strip whitespace, preserving file order
            memory_ids = [line.strip() for line in f if line.strip()]
        return memory_ids
    except Exception as e:
        console_logger.error(f"Error reading Memory ID file '{filepath}': {e}", exc_info=True)
        return []


def main():
    """Main execution function."""
    console_logger.info("--- Step 2: Manual L1 Knowledge Graph Construction Script Started ---")

    # Initialize L1 core components
    console_logger.info("Initializing L1 core components (MemoryGraph, L0Loader)...")
    try:
        l1_config = MemoryConfig(
            llm=settings.L1_LLM_CONFIG,
            embedder=settings.L1_EMBEDDER_CONFIG,
            graph_store=settings.L1_GRAPH_STORE_CONFIG
        )
        memory_graph_singleton = MemoryGraph(config=l1_config)
        l0_loader = L0MemoryLoader()
    except Exception as e:
        console_logger.critical(f"Failed to initialize L1 components, script cannot proceed: {e}", exc_info=True)
        return

    # Load pending tasks from local file
    memory_id_filepath = os.path.join(DATA_WORKING_DIRECTORY, MEMORY_ID_FILENAME)

    console_logger.info(f"Loading pending tasks sequentially from local file '{memory_id_filepath}'...")
    tasks_to_process = load_memory_ids_from_file(memory_id_filepath)

    if not tasks_to_process:
        console_logger.info("No pending L1 tasks found. Script completed.")
        return

    total_tasks = len(tasks_to_process)
    console_logger.info(f"Found {total_tasks} pending L1 tasks.")

    # Process tasks sequentially
    console_logger.info("\n--- Starting sequential processing of tasks to build Knowledge Graph ---")

    with tqdm(total=total_tasks, desc="Building L1 Knowledge Graph") as pbar:
        for memory_id in tasks_to_process:
            memory_data = l0_loader.get_memory_by_id(memory_id)
            if not memory_data:
                pbar.set_description(f"Skipping: Failed to load {memory_id}")
                pbar.update(1)
                continue

            try:
                pbar.set_description(f"Processing: {memory_id}")
                filters = {
                    "user_id": memory_data.get("user_id"),
                    "role_id": memory_data.get("role_id")
                }
                memory_graph_singleton.add(input_data=memory_data, filters=filters)
            except Exception as e:
                console_logger.error(f"L1 construction failed for memory_id '{memory_id}': {e}", exc_info=True)

            pbar.update(1)

    console_logger.info("\n--- L1 Construction Task Summary ---")
    console_logger.info(f"All {total_tasks} L0 memories listed in the file have been processed sequentially.")
    console_logger.info(f"Knowledge Graph has been updated.")
    console_logger.info("--- Script Execution Completed ---")


if __name__ == "__main__":
    main()