# scripts/1_produce_l0_memories.py

import os
import sys
import json
import threading
from tqdm import tqdm
from l0_memory.utils.logger import file_logger as logger

# --- Setup Python Path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# --- Core Module Imports ---
from config import settings
from l0_memory.core.memory_processor import MemoryProcessor
from l0_memory.utils.logger import console_logger

# ==============================================================================
#  >>>>> Configuration <<<<<
# ==============================================================================
LOCAL_HISTORY_FILE_PATH = ""
BATCH_SIZE = settings.TOPIC_ANALYSIS_BATCH_SIZE or 16

# Define the output filename for memory_ids
MEMORY_ID_FILENAME = "l0_memory_ids.txt"

# ==============================================================================
#  >>>>> Core Implementation <<<<<
# ==============================================================================

class LocalMemoryIdStore:
    """
    A simplified, thread-safe local file client responsible solely for
    appending memory_ids to a file.
    """

    def __init__(self, output_filepath: str):
        self.output_filepath = output_filepath
        self._lock = threading.Lock()
        # Clear the old file on initialization to ensure a fresh start
        with open(self.output_filepath, 'w') as f:
            f.write('')
        console_logger.info(f"Memory IDs will be saved to: {self.output_filepath}")

    def push_memory_id(self, memory_id: str):
        """Appends a memory_id as a new line to the file."""
        if not memory_id:
            return
        try:
            with self._lock:
                with open(self.output_filepath, 'a', encoding='utf-8') as f:
                    f.write(memory_id + '\n')
        except Exception as e:
            logger.error(f"Failed to append memory_id '{memory_id}' to local file '{self.output_filepath}': {e}", exc_info=True)


def load_history_from_file(filepath: str) -> list:
    """Loads dialogue history from a JSON or JSONL file."""
    if not os.path.exists(filepath):
        console_logger.error(f"Error: Input file {filepath} not found.")
        return []

    all_turns = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # Try parsing as JSONL first
            try:
                lines = content.strip().split('\n')
                for line in lines:
                    cleaned_line = line.strip()
                    if cleaned_line.endswith(','):
                        cleaned_line = cleaned_line[:-1]
                    if cleaned_line:
                        all_turns.append(json.loads(cleaned_line))
            except json.JSONDecodeError:
                # Fallback to parsing the entire file as a JSON array
                console_logger.info("JSONL parsing failed, attempting to parse as standard JSON array...")
                all_turns = json.loads(content)

        console_logger.info(f"Successfully loaded {len(all_turns)} dialogue records from {filepath}.")
        return sorted(all_turns, key=lambda x: x.get('created_at', ''))
    except Exception as e:
        console_logger.error(f"Unknown error while loading or parsing file {filepath}: {e}")
        return []


def group_dialogues_by_user(all_dialogues: list) -> dict:
    """Groups all dialogues by (user_id, role_id) pairs."""
    user_dialogues = {}
    for turn in tqdm(all_dialogues, desc="Grouping Dialogues by User"):
        user_id = turn.get('user_id')
        role_id = turn.get('role_id')
        if not all([user_id, role_id]):
            continue
        try:
            role_id = int(role_id)
        except (ValueError, TypeError):
            continue

        key = (user_id, role_id)
        if key not in user_dialogues:
            user_dialogues[key] = []
        user_dialogues[key].append(turn)
    return user_dialogues


def process_dialogue_block_for_l0(
        dialogues: list,
        user_id: str,
        role_id: int,
        l0_processor: MemoryProcessor,
        memory_id_store: LocalMemoryIdStore
):
    """
    Processes dialogues using a sliding window to generate L0 memories,
    and saves memory_ids to a local file.
    """
    if not dialogues:
        return

    console_logger.info(f"Generating L0 memories for {len(dialogues)} dialogues of ({user_id}, {role_id})...")

    # Add a virtual future turn to ensure the last chunk is processed
    last_turn = dialogues[-1]
    virtual_future_turn = last_turn.copy()
    virtual_future_turn["content"] = "End of history,goodbye."
    virtual_future_turn["created_at"] = "2100-01-01T00:00:00.000000+00:00"
    dialogues.append(virtual_future_turn)

    # Rewrite the call to process_chunk to capture memory_id manually
    original_process_chunk = l0_processor.process_chunk

    def hooked_process_chunk(dialogue_chunk, user_id, role_id):
        # Temporarily replace redis_publisher inside l0_processor
        original_publisher = l0_processor.redis_publisher

        class MockPublisher:
            def publish_memory_id(self, memory_id):
                memory_id_store.push_memory_id(memory_id) # Call local file store

        l0_processor.redis_publisher = MockPublisher()

        # Call original method
        result = original_process_chunk(dialogue_chunk, user_id, role_id)

        # Restore original publisher
        l0_processor.redis_publisher = original_publisher
        return result

    current_start_index = 0
    with tqdm(total=len(dialogues) - 1, desc=f"L0 Processing ({user_id})", leave=False) as pbar:
        while current_start_index < len(dialogues) - 1:
            chunk_to_process = dialogues[current_start_index: current_start_index + BATCH_SIZE]
            if not chunk_to_process:
                break

            # Use our hooked function
            last_processed_ts = hooked_process_chunk(chunk_to_process, user_id, role_id)

            # Sliding window logic
            last_processed_index_in_chunk = -1
            if last_processed_ts:
                for i in range(len(chunk_to_process) - 1, -1, -1):
                    if chunk_to_process[i]['created_at'] == last_processed_ts:
                        last_processed_index_in_chunk = i
                        break
            if last_processed_index_in_chunk != -1:
                processed_in_this_step = last_processed_index_in_chunk + 1
            else:
                processed_in_this_step = max(1, BATCH_SIZE - 5)
            pbar.update(processed_in_this_step)
            current_start_index += processed_in_this_step


def main():
    """Main execution function."""
    console_logger.info("--- Step 1: Offline L0 Memory Generation Script Started ---")

    # Initialize local file store instead of Redis queue
    history_dir = os.path.dirname(os.path.abspath(LOCAL_HISTORY_FILE_PATH))
    output_filepath = os.path.join(history_dir, MEMORY_ID_FILENAME)
    memory_id_store = LocalMemoryIdStore(output_filepath=output_filepath)

    # Initialize L0 processor
    l0_processor = MemoryProcessor()

    # Load and group dialogues
    all_dialogues = load_history_from_file(LOCAL_HISTORY_FILE_PATH)
    if not all_dialogues: return
    user_dialogue_groups = group_dialogues_by_user(all_dialogues)
    console_logger.info(f"Dialogue history divided into {len(user_dialogue_groups)} session groups.")

    # Process
    console_logger.info("\n--- Starting processing for all user groups to generate L0 memories ---")
    for (user_id, role_id), dialogues in user_dialogue_groups.items():
        process_dialogue_block_for_l0(dialogues, user_id, role_id, l0_processor, memory_id_store)

    console_logger.info("\n--- L0 Generation Task Summary ---")
    console_logger.info(f"L0 memories for all dialogue groups generated and stored in Elasticsearch.")
    console_logger.info(f"All generated memory_ids saved to file: {output_filepath}")
    console_logger.info(f"Next step: Please run '2_consume_and_build_l1.py' to build the Knowledge Graph.")
    console_logger.info("--- Script Execution Completed ---")


if __name__ == "__main__":
    main()