# l0_memory/utils/redis_publisher.py

import redis
from config import settings
from l0_memory.utils.logger import file_logger as logger


class RedisPublisher:
    """A simple Redis client for publishing messages."""

    def __init__(self, channel_name: str = "l1_memory_tasks"):
        try:
            # Reuse Redis configuration from settings.py
            self.redis_client = redis.Redis(**settings.REDIS_CONFIG)
            self.channel_name = channel_name
            self.redis_client.ping() # Optional: ping can be removed if startup speed is critical
            logger.info(f"RedisPublisher successfully connected to Redis, will publish to channel '{self.channel_name}'.")
        except Exception as e:
            logger.error(f"Failed to initialize RedisPublisher: {e}", exc_info=True)
            raise

    def publish_memory_id(self, memory_id: str):
        """
        Pushes a memory_id to a user-specific task queue and publishes a notification
        to the main channel.
        """
        if not memory_id:
            logger.warning("Attempted to publish an empty memory_id. Operation skipped.")
            return

        try:
            # 1. Parse user_session_key from the memory_id
            parts = memory_id.split('_')
            if len(parts) < 3:
                logger.error(f"Invalid memory_id format '{memory_id}', cannot parse user_session_key.")
                return
            user_id = '_'.join(parts[:-2])
            role_id = parts[-2]
            user_session_key = f"{user_id}_{role_id}"

            # 2. Get the key for the L1 task queue from templates
            queue_key = settings.REDIS_KEY_TEMPLATES['l1_task_queue'].format(user_session_key=user_session_key)

            # 3. Push the memory_id into the user's queue
            self.redis_client.rpush(queue_key, memory_id)

            # 4. Publish the user_session_key as a notification to the main channel
            num_clients = self.redis_client.publish(self.channel_name, user_session_key)

            logger.info(
                f"Successfully pushed memory_id '{memory_id}' to queue '{queue_key}' "
                f"and published notification '{user_session_key}' to channel '{self.channel_name}'. "
                f"Notified {num_clients} consumers."
            )
        except Exception as e:
            logger.error(f"Failed to push L1 task '{memory_id}' to Redis: {e}", exc_info=True)