# l0_memory/history_db/base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

class BaseHistoryDB(ABC):
    """
    历史对话数据库的抽象基类。
    [V10 修改] 查询接口使用 user_id 和 role_id。
    """

    @abstractmethod
    def get_dialogues_between(
        self,
        user_id: str,  # 【修改】
        role_id: int,   # 【修改】
        start_marker_created_at: Optional[str],
        end_marker_created_at: Optional[str],
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        获取指定 user_id 和 role_id 在两个时间戳之间的所有对话记录。
        """
        pass

    @abstractmethod
    def get_dialogue_count_between(
        self,
        user_id: str,  # 【修改】
        role_id: int,   # 【修改】
        start_marker_created_at: Optional[str],
        end_marker_created_at: Optional[str]
    ) -> int:
        """
        获取指定 user_id 和 role_id 在两个时间戳之间的对话记录数量。
        """
        pass