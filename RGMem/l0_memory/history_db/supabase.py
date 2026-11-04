# l0_memory/history_db/supabase.py

from typing import List, Dict, Optional, Any
from supabase import create_client, Client

from l0_memory.history_db.base import BaseHistoryDB
from config import settings
from l0_memory.utils.logger import file_logger as logger


class SupabaseHistoryDB(BaseHistoryDB):
    """
    Supabase (PostgreSQL) implementation of the historical dialogue database.
    """

    def __init__(self):
        self.supabase_client: Optional[Client] = None
        self.table_name = settings.SUPABASE_CONFIG.get("table_name")
        self.deploy_env = settings.SUPABASE_CONFIG.get("deploy_env")

        url = settings.SUPABASE_CONFIG.get("url")
        key = settings.SUPABASE_CONFIG.get("key")

        if not all([url, key, self.table_name]):
            logger.error("[SupabaseHistoryDB] Supabase URL, Key, or table name not fully provided in configuration.")
            raise ValueError("Incomplete Supabase configuration.")

        try:
            self.supabase_client: Client = create_client(url, key)
            log_message = f"[SupabaseHistoryDB] Initialized, using table: {self.table_name}"

            if self.deploy_env:
                log_message += f", filtering by deploy_env='{self.deploy_env}'"
            logger.info(log_message)
        except Exception as e:
            logger.error(f"[SupabaseHistoryDB] Failed to initialize Supabase client: {e}", exc_info=True)
            raise

    def get_dialogues_between(
            self,
            user_id: str,
            role_id: int,
            start_marker_created_at: Optional[str],
            end_marker_created_at: Optional[str],
            limit: int
    ) -> List[Dict[str, Any]]:
        """
        Retrieves all dialogue records for a specific user_id and role_id 
        between two timestamps from Supabase.
        """
        session_key = f"{user_id}_{role_id}"
        try:
            # Build query using chain calls
            query = self.supabase_client.from_(self.table_name).select("*")

            # Apply filters for user identity
            query = query.eq("user_id", user_id)
            query = query.eq("role_id", role_id)
            query = query.in_("role", ["user", "assistant"])

            if self.deploy_env:
                query = query.eq("deploy_env", self.deploy_env)

            # Apply time range filters
            if start_marker_created_at:
                # 'gt' (greater than)
                query = query.gt("created_at", start_marker_created_at)
            if end_marker_created_at:
                # 'lte' (less than or equal to)
                query = query.lte("created_at", end_marker_created_at)

            # Apply sorting and result limits
            query = query.order("created_at", desc=False).limit(limit)

            # Execute the query
            response = query.execute()

            return response.data if response.data else []

        except Exception as e:
            logger.error(
                f"[SupabaseHistoryDB] Failed to retrieve dialogues from Supabase (session_key: {session_key}): {e}",
                exc_info=True)
            return []

    def get_dialogue_count_between(
            self,
            user_id: str,
            role_id: int,
            start_marker_created_at: Optional[str],
            end_marker_created_at: Optional[str]
    ) -> int:
        """
        Retrieves the count of dialogue records for a specific user_id and role_id 
        between two timestamps from Supabase.
        """
        session_key = f"{user_id}_{role_id}"
        try:
            # Use count parameter for efficient counting without fetching all data
            query = self.supabase_client.from_(self.table_name).select(
                "*",
                count='exact'  # 'exact' returns the precise total count
            )

            # Apply filters
            query = query.eq("user_id", user_id)
            query = query.eq("role_id", role_id)
            query = query.in_("role", ["user", "assistant"])

            if self.deploy_env:
                query = query.eq("deploy_env", self.deploy_env)

            if start_marker_created_at:
                query = query.gt("created_at", start_marker_created_at)
            if end_marker_created_at:
                query = query.lte("created_at", end_marker_created_at)

            # Execute the query
            response = query.execute()

            return response.count if response.count is not None else 0

        except Exception as e:
            logger.error(
                f"[SupabaseHistoryDB] Failed to retrieve dialogue count from Supabase (session_key: {session_key}): {e}",
                exc_info=True)
            return 0