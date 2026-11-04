# l1_memory/l1_listener.py

import redis
import time
import concurrent.futures
import threading
from tenacity import retry, stop_after_attempt, wait_fixed
from l1_memory.configs.base import MemoryConfig
from l1_memory.graphs.neo4j import MemoryGraph
from l1_memory.data_loader import L0MemoryLoader
from config import settings
from l1_memory.utils import logger


class L1TaskListener:
    """
    A long-running service that listens to Redis for new memory_ids and processes them concurrently using a thread pool.
    """

    def __init__(self, memory_graph: MemoryGraph):
        logger.info("Initializing L1TaskListener...")
        try:
            self.graph_builder = memory_graph
            self.l0_loader = L0MemoryLoader()

            self.max_workers = settings.L1_LISTENER_MAX_WORKERS
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)

            self.redis_client = redis.Redis(**settings.REDIS_CONFIG)
            self.channel_name = "l1_memory_tasks"
            self._stop_event = threading.Event()

            logger.info(f"L1TaskListener initialized successfully. Thread pool size: {self.max_workers}")
        except Exception as e:
            logger.critical("L1TaskListener initialization failed.", exc_info=True)
            raise

    def _process_single_memory_with_retry(self, memory_id: str, user_session_key: str):
        """
        Encapsulates the processing logic for a single memory_id with tenacity retry mechanism.
        """

        @retry(stop=stop_after_attempt(3), wait=wait_fixed(5), reraise=True)
        def process_attempt():
            logger.info(f"[{user_session_key}] Processing memory_id: {memory_id}")
            memory_data = self.l0_loader.get_memory_by_id(memory_id)
            if not memory_data:
                # If data loading fails, it should also trigger a retry
                raise ValueError(f"Failed to load data for memory_id '{memory_id}' from ES.")

            filters = {
                "user_id": memory_data.get("user_id"),
                "role_id": memory_data.get("role_id")
            }
            self.graph_builder.add(input_data=memory_data, filters=filters)
            logger.info(f"[✓] [{user_session_key}] Memory ID '{memory_id}' processed successfully.")

        try:
            process_attempt()
        except Exception as e:
            logger.error(f"[✗] [{user_session_key}] Memory ID '{memory_id}' failed after max retries.", exc_info=True)
            # Move failed task to Dead Letter Queue
            dlq_key = settings.REDIS_KEY_TEMPLATES['l1_dead_letter_queue'].format(user_session_key=user_session_key)
            self.redis_client.rpush(dlq_key, memory_id)
            logger.warning(f"[{user_session_key}] Failed memory_id '{memory_id}' moved to DLQ '{dlq_key}'.")

    def _process_task(self, user_session_key: str):
        """
        Processes all pending tasks for a single user_session_key.
        Includes acquiring lock, processing the queue in a loop, and releasing the lock.
        """
        logger.info(f"[*] Received notification, preparing to process task queue for user '{user_session_key}'.")

        lock_key = settings.REDIS_KEY_TEMPLATES['l1_graph_lock'].format(user_session_key=user_session_key)

        # 1. Attempt to acquire L1 exclusive lock, timeout set to 10 minutes
        is_lock_acquired = self.redis_client.set(lock_key, "processing", nx=True, ex=600)

        if not is_lock_acquired:
            logger.info(f"[{user_session_key}] Failed to acquire lock, another thread is processing. Skipping notification.")
            return

        try:
            logger.info(f"[{user_session_key}] Lock acquired, starting to process task queue...")
            queue_key = settings.REDIS_KEY_TEMPLATES['l1_task_queue'].format(user_session_key=user_session_key)

            # 2. Process all tasks for this user until the queue is empty
            while memory_id := self.redis_client.lpop(queue_key):
                if isinstance(memory_id, bytes):
                    memory_id = memory_id.decode('utf-8')
                self._process_single_memory_with_retry(memory_id, user_session_key)

            logger.info(f"[{user_session_key}] Task queue processing completed.")

        except Exception as e:
            logger.error(f"[{user_session_key}] Unknown error occurred while processing task queue.", exc_info=True)
        finally:
            # 3. Ensure lock is released
            self.redis_client.delete(lock_key)
            logger.info(f"[{user_session_key}] Lock released.")

    def start_listening(self):
        """
        Connects to Redis and continuously listens to the task channel until stop() is called.
        """
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe(self.channel_name)
        logger.info(f"--- L1 Listener Service started, listening on Redis channel '{self.channel_name}' ---")

        while not self._stop_event.is_set():
            try:
                # Add timeout to check _stop_event periodically
                message = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message is None:
                    continue

                if message['type'] == 'message':
                    data = message['data']
                    if isinstance(data, bytes):
                        user_session_key = data.decode('utf-8')
                    else:
                        user_session_key = data
                    self.executor.submit(self._process_task, user_session_key)
            except Exception as e:
                # Catch exceptions in long-running service to prevent loop termination
                logger.error(f"Error in L1 listening loop: {e}", exc_info=True)
                time.sleep(5)

        logger.info("L1 Listener Service received stop signal, exiting loop.")
        pubsub.close()

    def stop(self):
        """Sets the stop event to gracefully terminate the listening loop."""
        logger.info("Sending stop signal to L1 Listener Service...")
        self._stop_event.set()

    def shutdown(self):
        """
        Sends stop signal first, then shuts down the thread pool.
        """
        self.stop()
        logger.info("Shutting down L1TaskListener thread pool...")
        self.executor.shutdown(wait=True)
        logger.info("L1TaskListener thread pool shut down.")