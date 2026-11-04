# l0_memory/history_db/__init__.py

from config import settings
from .base import BaseHistoryDB
from .elasticsearch import ElasticsearchHistoryDB
from .supabase import SupabaseHistoryDB
# 暂时在这里维护一个映射
DB_PROVIDERS = {
    "elasticsearch": ElasticsearchHistoryDB,
    "supabase": SupabaseHistoryDB,
    # "mysql": MySQLHistoryDB, # 未来可以这样扩展
}


def get_history_db_client() -> BaseHistoryDB:
    """
    数据库客户端工厂函数。
    根据 settings.py 中的配置，返回一个历史数据库的实例。
    """
    provider_name = settings.HISTORY_DB_PROVIDER
    if provider_name not in DB_PROVIDERS:
        raise ValueError(f"未知的历史数据库提供商: '{provider_name}'。请检查 config/settings.py。")

    # 从映射中获取类并实例化
    db_class = DB_PROVIDERS[provider_name]
    return db_class()