# l0_memory/rag_system/rag_server.py

from typing import List, Dict, Any, Optional
from l0_memory.utils.elasticsearch_client import ElasticsearchClient
from l0_memory.utils.logger import file_logger as logger
from config import settings
from l0_memory.rag_system.elasticsearch_retriever import ElasticsearchRetriever
from l0_memory.embedding_interface.factory import EmbeddingFactory
from l0_memory.llm_interface.factory import LlmFactory


class RagServer:
    """
    Handles the dual-track retrieval logic for RAG: Recency Memory + Relevance Memory.
    """

    def __init__(self):
        logger.info("--- RAG Server Initialization Started ---")
        self.es_client = ElasticsearchClient()

        # Dynamically create Embedding client using the Factory
        embedding_provider = settings.EMBEDDING_PROVIDER
        embedding_config_dict = settings.EMBEDDING_CONFIGS.get(embedding_provider)
        if not embedding_config_dict:
            raise ValueError(f"Configuration for RAG provider '{embedding_provider}' not found in EMBEDDING_CONFIGS.")
        self.embedding_client = EmbeddingFactory.create(embedding_provider, embedding_config_dict)

        # Dynamically create LLM client for RAG using the Factory
        provider = settings.LLM_PROVIDER
        provider_config = settings.L0_LLM_CONFIGS.get(provider)
        if not provider_config:
            raise ValueError(f"Configuration for provider '{provider}' not found in L0_LLM_CONFIGS.")

        rag_model_name = provider_config.get("rag_keyword_model")
        if not rag_model_name:
            # Fallback to worker_model if specific RAG model is not configured
            rag_model_name = provider_config["worker_model"]
            logger.warning(f"Dedicated rag_keyword_model not found, falling back to worker_model: {rag_model_name}")

        factory_config = {
            "model": rag_model_name,
            "api_key": provider_config.get("api_key"),
            "base_url": provider_config.get("base_url"),
            **provider_config.get("params", {})
        }

        llm_for_rag = LlmFactory.create(
            provider=provider,
            config=factory_config,
            client_name=f"RagServer{provider.capitalize()}Client"
        )

        self.retriever = ElasticsearchRetriever(
            es_client=self.es_client,
            embedding_client=self.embedding_client,
            llm_client=llm_for_rag,
            top_k=8
        )
        self.retriever.keyword_extraction_model = rag_model_name

        self.recency_size = settings.RAG_RECENCY_CONTEXT_SIZE
        logger.info(f"RAG Server initialization complete. Keyword extraction model: {rag_model_name}")

    def _format_docs_to_string(self, docs: List[Dict[str, Any]]) -> str:
        """Formats the retrieved atomic units into a readable string."""
        if not docs:
            return "None."

        formatted_strings = []
        for doc in docs:
            content = doc.get('content', '')
            unit_type = doc.get('unit_type', '')
            created_at = doc.get('parent_created_at', 'Unknown time')
            formatted_content = f"Recorded at {created_at}, {content}"
            prefix = "[Core Fact]: " if unit_type == 'core_fact' else "[Relevant Conclusion]: "
            formatted_strings.append(prefix + formatted_content)
        return "\n\n---\n\n".join(formatted_strings)

    def get_recency_context(self, user_id: str, role_id: int) -> List[Dict[str, Any]]:
        """Retrieves the latest N core_facts from L0 memories for immediate context."""
        session_key = f"{user_id}_{role_id}"
        try:
            response = self.es_client.es.search(
                index=settings.ES_INDEX_TOPICAL_L0,
                size=self.recency_size,
                query={
                    "bool": {
                        "filter": [
                            {"term": {"metadata.user_id": user_id}},
                            {"term": {"metadata.role_id": role_id}}
                        ]
                    }
                },
                sort=[{"metadata.created_at": "desc"}],
                _source=["atomic_units", "metadata.created_at"]
            )

            recency_core_facts = []
            for hit in response.get('hits', {}).get('hits', []):
                source = hit.get('_source', {})
                atomic_units = source.get('atomic_units', [])
                parent_created_at = source.get('metadata', {}).get('created_at', '')
                for unit in atomic_units:
                    if unit.get('unit_type') == 'core_fact':
                        recency_unit = dict(unit)
                        recency_unit['parent_created_at'] = parent_created_at
                        recency_core_facts.append(recency_unit)
                        break  # Retrieve only one core_fact per document
            return recency_core_facts
        except Exception as e:
            logger.error(f"[{session_key}] Failed to retrieve recency memory: {e}", exc_info=True)
            return []

    def get_relevance_context(self, user_id: str, role_id: int, query: str, keywords: Optional[str] = None) -> Dict[
        str, List[Dict[str, Any]]]:
        """Retrieves core_facts and conclusions most relevant to the query."""
        session_key = f"{user_id}_{role_id}"
        try:
            documents_dict = self.retriever._get_relevant_documents(
                query,
                run_manager=None,
                user_id=user_id,
                role_id=role_id,
                keywords=keywords
            )
            # Extract metadata from each Document object
            return {
                "core_facts": [doc.metadata for doc in documents_dict.get("core_facts", [])],
                "conclusions": [doc.metadata for doc in documents_dict.get("conclusions", [])]
            }
        except Exception as e:
            logger.error(f"[{session_key}] Failed to retrieve relevance memory: {e}", exc_info=True)
            return {"core_facts": [], "conclusions": []}

    def retrieve_context(self, user_id: str, role_id: int, query: str, keywords: Optional[str] = None) -> str:
        """
        Executes relevance retrieval and returns formatted relevance context.
        Note: Combining recency and relevance context is handled by the L2 layer.
        """
        session_key = f"{user_id}_{role_id}"
        logger.info(f"\n{'=' * 20} L0 Relevance Retrieve Start {'=' * 20}")
        logger.info(f"Session Key: {session_key}")
        logger.info(f"Query: {query}")

        # 1. Fetch only relevant memory
        relevance_dict = self.get_relevance_context(user_id, role_id, query, keywords=keywords)

        # 2. Format relevance memory
        core_facts_str = self._format_docs_to_string(relevance_dict.get("core_facts", []))
        conclusions_str = self._format_docs_to_string(relevance_dict.get("conclusions", []))

        # 3. Return structured string containing relevance context
        relevance_context_str = f"""
Relevant Core Facts:
---
{core_facts_str}
 ---

Relevant Conclusions:
---
{conclusions_str}
---
"""
        logger.info(f"\n--- [L0 Relevance Context Only] ---\n{relevance_context_str.strip()}")
        logger.info(f"\n{'=' * 20} L0 Relevance Retrieve End {'=' * 20}")

        return relevance_context_str
