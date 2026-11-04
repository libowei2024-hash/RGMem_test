# l0_memory/rag_system/elasticsearch_retriever.py

import time
from typing import List, Dict, Any, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from l0_memory.embedding_interface.base import EmbeddingBase

from l0_memory.utils.elasticsearch_client import ElasticsearchClient
from l0_memory.utils.logger import file_logger as logger
from config import settings, new_prompts as prompts


class ElasticsearchRetriever(BaseRetriever):
    """
    Elasticsearch retriever for L0 memory, supporting hybrid search for atomic units (facts and conclusions).
    """
    es_client: ElasticsearchClient
    embedding_client: EmbeddingBase
    llm_client: Optional[Any] = None
    top_k: int = max(settings.RAG_CORE_FACT_TOP_K, settings.RAG_CONCLUSION_TOP_K)
    keyword_extraction_model: Optional[str] = None

    def _retrieve_atomic_units(
            self,
            query: str,
            extracted_keywords: str,
            query_vector: List[float],
            user_id_filters: List[Dict],
            unit_type: str,
            top_k: int,
            keyword_threshold: float,
            vector_threshold: float
    ) -> List[Document]:
        """
        Reusable function for retrieving specific types of atomic units.
        Includes extraction, filtering, fusion, and sorting logic.
        """
        # Prepare filters for the nested object
        nested_filters = [{"term": {"atomic_units.unit_type": unit_type}}]

        # Execute searches
        keyword_units, vector_units = [], []
        if settings.RAG_RETRIEVAL_MODE in ["hybrid", "keyword_only"]:
            raw_keyword_hits = self._execute_keyword_search(extracted_keywords, user_id_filters, nested_filters)
            for hit in raw_keyword_hits:
                parent_id = hit['_id']
                parent_created_at = hit.get('_source', {}).get('metadata', {}).get('created_at', 'Unknown time')
                for inner_hit in hit.get('inner_hits', {}).get('atomic_units', {}).get('hits', {}).get('hits', []):
                    unit = inner_hit['_source']
                    unit['_score'] = inner_hit['_score']
                    unit['_parent_id'] = parent_id
                    unit['parent_created_at'] = parent_created_at
                    keyword_units.append(unit)

        if settings.RAG_RETRIEVAL_MODE in ["hybrid", "vector_only"]:
            raw_vector_hits = self._execute_vector_search(query_vector, user_id_filters, nested_filters)
            for hit in raw_vector_hits:
                parent_id = hit['_id']
                parent_created_at = hit.get('_source', {}).get('metadata', {}).get('created_at', 'Unknown time')
                for inner_hit in hit.get('inner_hits', {}).get('atomic_units', {}).get('hits', {}).get('hits', []):
                    unit = inner_hit['_source']
                    unit['_score'] = inner_hit['_score']
                    unit['_parent_id'] = parent_id
                    unit['parent_created_at'] = parent_created_at
                    vector_units.append(unit)

        # Filtering and fusion
        filtered_keyword_units = [u for u in keyword_units if u.get('_score', 0) >= keyword_threshold]
        filtered_vector_units = [u for u in vector_units if u.get('_score', 0) >= vector_threshold]

        if settings.RAG_RETRIEVAL_MODE == "hybrid":
            sorted_keyword_units = sorted(filtered_keyword_units, key=lambda x: x.get('_score', 0), reverse=True)
            sorted_vector_units = sorted(filtered_vector_units, key=lambda x: x.get('_score', 0), reverse=True)
            fused_results = self._reciprocal_rank_fusion_units([sorted_keyword_units, sorted_vector_units])
            sorted_units = sorted(fused_results.values(), key=lambda x: x['rrf_score'], reverse=True)
        elif settings.RAG_RETRIEVAL_MODE == "keyword_only":
            sorted_units = sorted(filtered_keyword_units, key=lambda x: x.get('_score', 0), reverse=True)
        elif settings.RAG_RETRIEVAL_MODE == "vector_only":
            sorted_units = sorted(filtered_vector_units, key=lambda x: x.get('_score', 0), reverse=True)
        else:
            sorted_units = []

        # Build and return LangChain Document list
        documents = []
        seen_unit_ids = set()
        for unit_data in sorted_units[:top_k]:
            unit = unit_data.get('source', unit_data)
            unit_id = unit.get('unit_id')
            if unit_id and unit_id not in seen_unit_ids:
                documents.append(Document(page_content=unit.get('content', ''), metadata=unit))
                seen_unit_ids.add(unit_id)
        return documents

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun, user_id: str | None = None,
            role_id: int | None = None,
            keywords: Optional[str] = None
    ) -> Dict[str, List[Document]]:
        """
        High-level coordination function to retrieve relevant core_facts and conclusions separately.
        """
        session_key = f"{user_id}_{role_id}" if user_id and role_id else "GLOBAL"
        retrieval_log = f"\n--- RAG Relevance Search for Query: '{query}' [SessionKey: {session_key}] ---\n"

        extracted_keywords = ""
        query_vector = []

        if settings.RAG_RETRIEVAL_MODE in ["hybrid", "keyword_only"]:
            if keywords:
                extracted_keywords = keywords
                retrieval_log += f"\nUsing caller-provided keywords: {extracted_keywords}\n"
            else:
                retrieval_log += "\nNo keywords provided, extracting internally...\n"
                extracted_keywords = self._extract_keywords(query)
                retrieval_log += f"\nExtracted Keywords: {extracted_keywords}\n"

        if settings.RAG_RETRIEVAL_MODE in ["hybrid", "vector_only"]:
            query_vector = self.embedding_client.embed_query(query)

        # Prepare top-level document filters
        user_id_filters = []
        if user_id:
            user_id_filters.append({"term": {"metadata.user_id": user_id}})
        if role_id:
            user_id_filters.append({"term": {"metadata.role_id": role_id}})

        # Step 1: Retrieve Core Facts
        retrieval_log += "\n--- Retrieving Core Facts ---"
        core_facts_docs = self._retrieve_atomic_units(
            query=query,
            extracted_keywords=extracted_keywords,
            query_vector=query_vector,
            user_id_filters=user_id_filters,
            unit_type="core_fact",
            top_k=settings.RAG_CORE_FACT_TOP_K,
            keyword_threshold=settings.RAG_CORE_FACT_KEYWORD_THRESHOLD,
            vector_threshold=settings.RAG_CORE_FACT_VECTOR_THRESHOLD
        )
        retrieval_log += f"\nFound {len(core_facts_docs)} relevant core facts."

        # Step 2: Retrieve Conclusions
        retrieval_log += "\n\n--- Retrieving Conclusions ---"
        conclusions_docs = self._retrieve_atomic_units(
            query=query,
            extracted_keywords=extracted_keywords,
            query_vector=query_vector,
            user_id_filters=user_id_filters,
            unit_type="conclusion",
            top_k=settings.RAG_CONCLUSION_TOP_K,
            keyword_threshold=settings.RAG_CONCLUSION_KEYWORD_THRESHOLD,
            vector_threshold=settings.RAG_CONCLUSION_VECTOR_THRESHOLD
        )
        retrieval_log += f"\nFound {len(conclusions_docs)} relevant conclusions.\n"

        logger.info(retrieval_log)

        return {
            "core_facts": core_facts_docs,
            "conclusions": conclusions_docs
        }

    def _extract_keywords(self, query: str) -> str:
        """Uses LLM to extract search keywords from the query."""
        if not self.llm_client:
            logger.warning("LLM client not provided. Falling back to raw query for keyword search.")
            return query

        try:
            prompt = prompts.KEYWORD_EXTRACTION_PROMPT.format(query=query)
            if hasattr(self.llm_client, 'get_completion'):
                keywords = self.llm_client.get_completion(prompt, model_override=self.keyword_extraction_model)
            elif hasattr(self.llm_client, 'invoke'):
                response = self.llm_client.invoke(prompt)
                keywords = response.content if hasattr(response, 'content') else str(response)
            else:
                logger.warning(
                    f"llm_client of type {type(self.llm_client)} lacks recognized methods ('get_completion' or 'invoke').")
                return query

            return keywords.strip()
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}. Using raw query as fallback.", exc_info=True)
            return query

    def _execute_keyword_search(self, keywords: str, user_id_filters: List[Dict], nested_filters: List[Dict] = None) -> \
    List[Dict[str, Any]]:
        """Executes a nested keyword search in Elasticsearch."""
        if not keywords:
            return []
        nested_query_bool = {"must": [{"multi_match": {"query": keywords, "fields": ["atomic_units.content"]}}]}
        if nested_filters:
            nested_query_bool["filter"] = nested_filters

        query = {
            "bool": {
                "must": [{"nested": {"path": "atomic_units", "query": {"bool": nested_query_bool}, "inner_hits": {}}}],
                "filter": user_id_filters
            }
        }
        try:
            resp = self.es_client.es.search(index=settings.ES_INDEX_TOPICAL_L0, size=self.top_k * 2, query=query)
            return resp.get('hits', {}).get('hits', [])
        except Exception as e:
            logger.error(f"Keyword search failed: {e}", exc_info=True)
            return []

    def _execute_vector_search(self, query_vector: List[float], user_id_filters: List[Dict],
                               nested_filters: List[Dict] = None) -> List[Dict[str, Any]]:
        """Executes a nested kNN vector search."""
        all_filters = user_id_filters[:]
        if nested_filters:
            all_filters.append({
                "nested": {
                    "path": "atomic_units",
                    "query": {"bool": {"filter": nested_filters}}
                }
            })

        knn_query = {
            "field": "atomic_units.content_vector",
            "query_vector": query_vector,
            "k": self.top_k * 2,
            "num_candidates": 100,
            "filter": {"bool": {"filter": all_filters}},
            "inner_hits": {}
        }

        try:
            resp = self.es_client.es.search(
                index=settings.ES_INDEX_TOPICAL_L0,
                knn=knn_query,
                _source=False,
                size=self.top_k * 2
            )
            return resp.get('hits', {}).get('hits', [])
        except Exception as e:
            logger.error(f"Vector search failed: {e}", exc_info=True)
            return []

    def _reciprocal_rank_fusion_units(self, sorted_result_lists: List[List[Dict]], k: int = 60) -> Dict[str, Dict]:
        """Performs Reciprocal Rank Fusion (RRF) on multiple sorted result lists."""
        fused_scores = {}
        for unit_list in sorted_result_lists:
            for rank, unit in enumerate(unit_list):
                unit_id = unit['unit_id']
                if unit_id not in fused_scores:
                    fused_scores[unit_id] = {"source": unit, "rrf_score": 0}
                fused_scores[unit_id]['rrf_score'] += 1 / (k + rank + 1)
        return fused_scores

    def _format_log_section(self, title: str, hits: List[Dict], is_fused: bool = False) -> str:
        """Helper to format retrieval hits into a readable log string."""
        log_str = f"\n--- {title} ({len(hits)} hits) ---\n"
        if not hits:
            return log_str + "No results.\n"
        for hit in hits:
            if is_fused:
                score = hit.get('rrf_score', 0.0)
                unit = hit.get('source', {})
            else:
                score = hit.get('_score', 0.0)
                unit = hit
            content = unit.get('content', 'Unknown content')
            parent_id = unit.get('_parent_id', 'N/A')
            unit_id = unit.get('unit_id', 'N/A')
            log_line = f"  - Score: {score:.4f} | Parent: {parent_id} | Unit: {unit_id} | Content: {content}"
            log_str += log_line + "\n"
        return log_str