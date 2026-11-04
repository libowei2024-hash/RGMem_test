# l0_memory/memory_server.py

import concurrent.futures
from functools import partial
from config import settings

# Dynamically import the HistoryDB implementation based on settings
if settings.HISTORY_DB_PROVIDER == 'elasticsearch':
    from l0_memory.history_db.elasticsearch import ElasticsearchHistoryDB as HistoryDB
elif settings.HISTORY_DB_PROVIDER == 'supabase':
    from l0_memory.history_db.supabase import SupabaseHistoryDB as HistoryDB
else:
    raise ValueError(f"Unknown history database provider: {settings.HISTORY_DB_PROVIDER}")

from l0_memory.utils.logger import file_logger as logger
from l0_memory.redis_db.state_manager import RedisStateManager
from l0_memory.core.memory_processor import MemoryProcessor

class MemoryServer:
    """
    Manages the asynchronous processing of dialogue chunks into structured L0 memories.
    """

    def __init__(self):
        logger.info("--- Memory Server Initialization Started ---")
        self.redis_manager = RedisStateManager()
        self.history_db = HistoryDB()
        self.memory_processor = MemoryProcessor()
        self.max_workers = settings.MEMORY_SERVER_MAX_WORKERS
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.k_threshold = settings.TOPIC_ANALYSIS_BATCH_SIZE
        logger.info(f"Server initialization complete. Background workers: {self.max_workers}, Trigger threshold k={self.k_threshold}")

    def on_task_completed(self, user_id: str, role_id: int, last_dialogue_in_chunk: dict,
                          future: concurrent.futures.Future):
        """Callback function executed upon background task completion."""
        session_key = f"{user_id}_{role_id}"
        try:
            # The future result is the timestamp of the last processed topic's end
            last_processed_topic_end_ts = future.result()

            if last_processed_topic_end_ts:
                new_marker_ts = last_processed_topic_end_ts
            else:
                # Fallback to the timestamp of the last dialogue in the entire chunk if no topics were processed
                new_marker_ts = last_dialogue_in_chunk[-1]['created_at']

            self.redis_manager.update_processed_marker(user_id, role_id, new_marker_ts)

        except Exception as e:
            logger.error(f"[{session_key}] Background task execution failed: {e}", exc_info=True)
        finally:
            self.redis_manager.release_session_lock(user_id, role_id)

    def trigger_memory_processing(self, user_id: str, role_id: int, created_at: str):
        """Main entry point for the service, triggered by new messages."""
        session_key = f"{user_id}_{role_id}"

        if not self.redis_manager.acquire_session_lock(user_id, role_id):
            logger.info(f"[{session_key}] Failed to acquire lock, a task is already in progress.")
            return

        try:
            state = self.redis_manager.update_and_get_session_state(user_id, role_id, created_at)
            start_ts = state.get('last_processed_created_at')
            end_ts = state.get('latest_incoming_created_at')

            dialogue_count = self.history_db.get_dialogue_count_between(user_id, role_id, start_ts, end_ts)
            logger.info(f"[{session_key}] Found {dialogue_count} new messages between ({start_ts}, {end_ts}].")

            if dialogue_count >= self.k_threshold:
                logger.info(f"[{session_key}] New message count reached threshold {self.k_threshold}, submitting processing task.")

                dialogues_to_process = self.history_db.get_dialogues_between(
                    user_id, role_id, start_ts, end_ts, limit=dialogue_count
                )

                if not dialogues_to_process:
                    self.redis_manager.release_session_lock(user_id, role_id)
                    return

                future = self.executor.submit(
                    self.memory_processor.process_chunk, dialogues_to_process, user_id, role_id
                )
                callback = partial(self.on_task_completed, user_id, role_id, dialogues_to_process)
                future.add_done_callback(callback)
            else:
                self.redis_manager.release_session_lock(user_id, role_id)

        except Exception as e:
            logger.error(f"[{session_key}] An error occurred in the main trigger process: {e}", exc_info=True)
            self.redis_manager.release_session_lock(user_id, role_id)

    def shutdown(self):
        """Shuts down the server gracefully, waiting for all tasks to complete."""
        logger.info("--- Memory Server Shutting Down ---")
        self.executor.shutdown(wait=True)
        logger.info("All background tasks have completed. Server shut down.")