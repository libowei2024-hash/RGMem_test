# config/settings.py

import os
from dotenv import load_dotenv

load_dotenv()

# ==================== LLM Provider Settings ====================
# Options: 'volc', 'bailian', 'openai', 'gemini'
LLM_PROVIDER = "openai"

L0_LLM_CONFIGS = {
    "gemini_proxy": {
        "api_key": os.getenv("GEMINI_API_KEY", ""),
        "base_url": "http://gemini.example-proxy.com/v1",
        "worker_model": "gemini-2.5-flash",
        "rag_keyword_model": "gemini-2.5-flash",
        "task_models": {
            "persona_model": "gemini-2.5-flash",
        },
        "params": {
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 8192
        }
    },
    "volc": {
        "api_key": os.getenv("ARK_API_KEY"),
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "worker_model": "doubao-seed-1-6-thinking-250715",
        "rag_keyword_model": "doubao-seed-1.6-250615",
        "task_models": {
            "persona_model": "doubao-seed-1-6-thinking-250715",
            "judge_model": "deepseek-r1-250528",
            "eval_chat_model": "doubao-seed-1-6-thinking-250715",
        },
        "params": {
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 8192
        }
    },
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "base_url": "https://api.openai.com/v1",
        "worker_model": "gpt-4.1",
        "rag_keyword_model": "gpt-6.1-sol",
        "task_models": {
            "persona_model": "gpt-4-turbo",
            "judge_model": "gpt-4.1-mini",
            "eval_chat_model": "gpt-6.1-sol",
        },
        "params": {
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 16000
        }
    },
    "bailian": {
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "worker_model": "qwen-max",
        "rag_keyword_model": "qwen-plus",
        "params": {
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 4096
        }
    },
    "gemini": {
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "base_url": "https://api.openai-proxy.org/google",
        "worker_model": "gemini-1.5-flash-latest",
        "rag_keyword_model": "gemini-1.5-flash-latest",
        "task_models": {
            "persona_model": "gemini-1.5-pro-latest",
            "judge_model": "gemini-1.5-pro-latest",
            "eval_chat_model": "gemini-1.5-pro-latest",
        },
        "params": {
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 8192
        }
    }
}

# ==================== Elasticsearch Settings ====================
ES_INDEX_TOPICAL_L0 = "evaluation_persona_mem"
ES_INDEX_CHAT_HISTORY = "permanent_chat_log_v2"

# ==================== Memory Parameters ====================
TOPIC_ANALYSIS_BATCH_SIZE = 16    # Batch size for triggering topic segmentation analysis
MAX_BUFFER_SIZE = 5000            # Max dialogue buffer size to prevent infinite growth
MEMORY_SERVER_MAX_WORKERS = 10    # Thread pool size for concurrent memory generation
PERSONA_GENERATION_THREADS = 8    # Thread pool size for concurrent persona generation

# Default start timestamp for new session states.
# Use ISO 8601 format (e.g., "2024-08-01T00:00:00.000000+00:00")
DEFAULT_PROCESSING_START_TIME = "2025-08-25T00:00:00.000000+00:00"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==================== RAG Settings ====================
# Options: 'hybrid', 'keyword_only', 'vector_only'
RAG_RETRIEVAL_MODE = "keyword_only"
RAG_RECENCY_CONTEXT_SIZE = 5          # Number of latest L0 memories for context

# Core-fact parameters
RAG_CORE_FACT_TOP_K = 5
RAG_CORE_FACT_KEYWORD_THRESHOLD = 0
RAG_CORE_FACT_VECTOR_THRESHOLD = 0.3

# Conclusions parameters
RAG_CONCLUSION_TOP_K = 30
RAG_CONCLUSION_KEYWORD_THRESHOLD = 1.0
RAG_CONCLUSION_VECTOR_THRESHOLD = 0.3

# ==================== Embedding Model Settings ====================
# Options: 'local', 'volc', 'gemini', 'openai'
EMBEDDING_PROVIDER = "openai"

EMBEDDING_CONFIGS = {
    "local": {
        "model": "BAAI/bge-small-zh-v1.5",
        "encode_kwargs": {"normalize_embeddings": True}
    },
    "volc": {
        "model": "doubao-embedding-text-240715",
        "api_key": os.getenv("ARK_API_KEY"),
        "dimensions": 2560
    },
    "gemini": {
        "model": "models/text-embedding-004",
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "dimensions": 768,
        "base_url": os.getenv("GEMINI_PROXY_URL")
    },
    "openai": {
        "model": "text-embedding-3-large",
        "api_key": os.getenv("OPENAI_EMBEDDING_KEY", "REDACTED_API_KEY"),
        "dimensions": 3072,
        "base_url": "https://api.openai.com/v1"
    }
}

# ==================== Database Settings ====================
# Options: 'elasticsearch', 'supabase'
HISTORY_DB_PROVIDER = "supabase"

# Redis Configuration
REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'your-redis-endpoint.com'),
    'port': int(os.getenv('REDIS_PORT', 6379)),
    'password': os.getenv('REDIS_PASSWORD', 'REDACTED_PASSWORD'),
    'db': int(os.getenv('REDIS_DB', 0)),
    'decode_responses': True
}

# Redis Key Templates
REDIS_KEY_TEMPLATES = {
    'session_state': "session:{session_id}:state",
    'session_lock': "session:{session_id}:lock",
    'l1_graph_lock': "l1_graph:{user_session_key}:lock",
    'l1_task_queue': "l1_queue:{user_session_key}",
    'l1_dead_letter_queue': "l1_dlq:{user_session_key}",
    'user_system_prompt_content': "user_id:{user_id}:chat_biz:system_prompt_content",
    'user_previous_response_id': "user_id:{user_id}:chat_biz:user_previous_response_id"
}

# Supabase Configuration
SUPABASE_CONFIG = {
    "url": os.getenv("SUPABASE_URL"),
    "key": os.getenv("SUPABASE_KEY"),
    "table_name": os.getenv("SUPABASE_CHAT_LOG_TABLE", "messages"),
    "deploy_env": "verify" # Env filtering: 'preprod', 'test', 'verify'
}

# Elasticsearch Configuration
ES_CONFIG = {
    "host": os.getenv("ES_HOST", "https://localhost:9200"),
    "user": os.getenv("ES_USER", "elastic"),
    "password": os.getenv("ES_PASSWORD", "REDACTED_PASSWORD")
}

# ==================== L1 Memory Settings ====================
inference_threshold = 3
summary_threshold = 6
node_similar_threshold = 0.75
retrieval_similar_threshold = 0.7
vector_search_limit_per_entity = 5
retrieval_top_k = 3
L1_LISTENER_MAX_WORKERS = 8

# L1 Memory - LLM Configuration
L1_VOLCANO_LLM_CONFIG = {
    "provider": "volcano",
    "config": {
        "model": "doubao-seed-1-6-thinking-250715",
        "temperature": 0.5,
        "max_tokens": 32768,
    }
}

L1_GEMINI_LLM_CONFIG = {
    "provider": "gemini",
    "config": {
        "model": "gemini-1.5-pro-latest",
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "openai_base_url": os.getenv("GEMINI_PROXY_URL"),
        "temperature": 0.4,
        "max_tokens": 8192,
        "top_p": 0.9,
    }
}

L1_OPENAI_LLM_CONFIG = {
    "provider": "openai",
    "config": {
        "model": "gpt-4.1",
        "api_key": os.getenv("L1_OPENAI_API_KEY", "REDACTED_API_KEY"),
        "openai_base_url": "https://api.openai.com/v1",
        "temperature": 0.3,
        "max_tokens": 8192,
    }
}

L1_LLM_CONFIG = L1_OPENAI_LLM_CONFIG

# L1 Memory - Embedder Configuration
L1_VOLCANO_EMBEDDER_CONFIG = {
    "provider": "volcano",
    "config": {
        "model": "doubao-embedding-text-240715",
        "api_key": os.getenv("ARK_API_KEY"),
        "embedding_dims": 2560
    }
}

L1_GEMINI_EMBEDDER_CONFIG = {
    "provider": "gemini",
    "config": {
        "model": "models/text-embedding-004",
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "embedding_dims": 768,
        "openai_base_url": os.getenv("GEMINI_PROXY_URL"),
    }
}

L1_OPENAI_EMBEDDER_CONFIG = {
    "provider": "openai",
    "config": {
        "model": "text-embedding-3-small",
        "api_key": os.getenv("L1_OPENAI_EMBEDDING_KEY", "REDACTED_API_KEY"),
        "embedding_dims": 1536,
        "openai_base_url": "https://api.openai.com/v1",
    }
}

L1_EMBEDDER_CONFIG = L1_OPENAI_EMBEDDER_CONFIG

# L1 Memory - Graph Store (Neo4j) Configuration
L1_GRAPH_STORE_CONFIG = {
    "provider": "neo4j",
    "config": {
        "url": os.getenv("NEO4J_URL", "neo4j://localhost:7687"),
        "username": os.getenv("NEO4J_USER", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "REDACTED_PASSWORD"),
        "database": "neo4j"
    }
}

# User Whitelist for Global Knowledge View
L1_USER_ID_WHITELIST = [
    "USER_ID_EXAMPLE_1",
    "USER_ID_EXAMPLE_2",
    "USER_ID_EXAMPLE_3"
]
