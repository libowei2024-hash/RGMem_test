# l0_memory/redis_db/state_manager.py

import redis
from config import settings
from l0_memory.utils.logger import file_logger as logger


class RedisStateManager:
    """
    Uses Redis to manage session processing states and distributed locks.
    Driven entirely by configurations in settings.py.
    """

    def __init__(self):
        try:
            self.redis_client = redis.Redis(**settings.REDIS_CONFIG)
            self.redis_client.ping()
            logger.info(f"Successfully connected to Redis: {settings.REDIS_CONFIG['host']}:{settings.REDIS_CONFIG['port']}")
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}", exc_info=True)
            raise

    def _get_key(self, key_template_name: str, **kwargs) -> str:
        """Helper function to format Redis keys based on templates."""
        return settings.REDIS_KEY_TEMPLATES[key_template_name].format(**kwargs)

    def _get_session_key(self, user_id: str, role_id: int) -> str:
        """Combines user_id and role_id into a unique string for Redis keys."""
        return f"{user_id}_{str(role_id)}"

    def get_session_state(self, user_id: str, role_id: int) -> dict:
        """
        Read-only operation: Retrieves the complete current state of a session.

        Returns:
            A dictionary containing the state, e.g.:
            {'latest_incoming_created_at': '...', 'last_processed_created_at': '...'}
            Returns an empty dictionary {} if the session does not exist.
        """
        session_key = self._get_session_key(user_id, role_id)
        state_key = self._get_key('session_state', session_id=session_key)

        current_state = self.redis_client.hgetall(state_key)
        return current_state

    def update_and_get_session_state(self, user_id: str, role_id: int, created_at: str) -> dict:
        """
        Atomically updates the latest message timestamp and retrieves the full session state.
        """
        session_key = self._get_session_key(user_id, role_id)
        state_key = self._get_key('session_state', session_id=session_key)

        pipeline = self.redis_client.pipeline()
        pipeline.hset(state_key, 'latest_incoming_created_at', created_at)
        pipeline.hsetnx(state_key, 'last_processed_created_at', settings.DEFAULT_PROCESSING_START_TIME)
        pipeline.hgetall(state_key)

        results = pipeline.execute()
        return results[2]

    def acquire_session_lock(self, user_id: str, role_id: int, lock_timeout_seconds: int = 500) -> bool:
        """
        Attempts to acquire a distributed lock for a specific user_id and role_id.
        """
        session_key = self._get_session_key(user_id, role_id)
        lock_key = self._get_key('session_lock', session_id=session_key)
        is_lock_acquired = self.redis_client.set(lock_key, "processing", nx=True, ex=lock_timeout_seconds)
        return bool(is_lock_acquired)

    def release_session_lock(self, user_id: str, role_id: int):
        """Releases the distributed lock for a specific user_id and role_id."""
        session_key = self._get_session_key(user_id, role_id)
        lock_key = self._get_key('session_lock', session_id=session_key)
        self.redis_client.delete(lock_key)
        logger.info(f"[{session_key}] Distributed lock released.")

    def update_processed_marker(self, user_id: str, role_id: int, new_processed_created_at: str):
        """
        Updates the session's processing marker to the latest timestamp.
        """
        session_key = self._get_session_key(user_id, role_id)
        state_key = self._get_key('session_state', session_id=session_key)
        self.redis_client.hset(state_key, 'last_processed_created_at', new_processed_created_at)
        logger.info(f"[{session_key}] Processing progress updated to timestamp: {new_processed_created_at}")