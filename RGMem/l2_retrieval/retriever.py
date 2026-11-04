# file: l2_retrieval/retriever.py

import asyncio
from typing import List, Dict, Tuple, Optional

from .l0_adapter import L0Adapter
from .l1_adapter import L1Adapter
from .formatter import format_context_as_markdown
from l0_memory.utils.logger import file_logger as logger

L2_RETRIEVAL_TIMEOUT = 120.0


class L2Retriever:
    """
    The unified L2 retrieval layer. It orchestrates parallel queries
    to L0 and L1 memory based on the specified retrieval type.
    """

    def __init__(self, l1_adapter: L1Adapter):
        """Initializes the adapters for both memory layers."""
        self.l0_adapter = L0Adapter()
        self.l1_adapter = l1_adapter
        logger.info("[L2 Retriever] Unified L2 Retriever initialized.")

    def _preprocess_input_for_l0_keywords(self, entity_queries: List[Dict]) -> str:
        """
        Processes the input query list to prepare the 'keywords' string for L0.
        """
        all_keywords = {q["name"] for q in entity_queries}
        l0_keywords_string = " ".join(sorted(list(all_keywords)))
        return l0_keywords_string

    async def retrieve(
            self,
            query: str,
            user_id: str,
            role_id: int,
            retrieval_type: str = "relevance",
            entity_queries: Optional[List[Dict]] = None,
            keywords: Optional[str] = None
    ) -> str:
        """
        Executes a unified retrieval, intelligently handling both new and legacy request formats.
        """
        logger.info(f"[L2 Retriever] Starting retrieval with type: '{retrieval_type}'")

        tasks = []
        task_map = {}

        l0_keywords_string = ""
        l1_queries = []

        if entity_queries:
            logger.info("Processing new format request with 'entity_queries', enabling L0+L1 retrieval.")
            l0_keywords_string = self._preprocess_input_for_l0_keywords(entity_queries)
            l1_queries = [q for q in entity_queries if q.get("retrieve")]
        elif keywords:
            logger.info("Processing legacy format request with 'keywords', enabling L0-only relevance retrieval.")
            l0_keywords_string = keywords
            l1_queries = []

        if retrieval_type in ["recency", "hybrid"]:
            recency_task = asyncio.to_thread(self.l0_adapter.retrieve_recency, user_id, role_id)
            tasks.append(recency_task)
            task_map["recency"] = len(tasks) - 1

        if retrieval_type in ["relevance", "hybrid"]:
            if l0_keywords_string:
                relevance_l0_task = asyncio.to_thread(
                    self.l0_adapter.retrieve_relevance,
                    user_id, role_id, query, l0_keywords_string
                )
                tasks.append(relevance_l0_task)
                task_map["relevance_l0"] = len(tasks) - 1

            if l1_queries:
                relevance_l1_task = asyncio.to_thread(self.l1_adapter.retrieve, user_id, role_id, l1_queries)
                tasks.append(relevance_l1_task)
                task_map["relevance_l1"] = len(tasks) - 1

        if not tasks:
            logger.warning("[L2 Retriever] No tasks scheduled for retrieval. Returning empty result.")
            return format_context_as_markdown(None, None, None, query)

        recency_data, relevance_l0_data, relevance_l1_data = None, None, None

        try:
            logger.info(
                f"[L2 Retriever] Executing {len(tasks)} tasks in parallel (timeout: {L2_RETRIEVAL_TIMEOUT}s)...")
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=L2_RETRIEVAL_TIMEOUT
            )

            if "recency" in task_map:
                recency_data = results[task_map["recency"]]
            if "relevance_l0" in task_map:
                relevance_l0_data = results[task_map["relevance_l0"]]
            if "relevance_l1" in task_map:
                relevance_l1_data = results[task_map["relevance_l1"]]

            # Exception handling for parallel tasks
            if isinstance(recency_data, Exception):
                logger.error("[L2 Retriever] Recency task failed.", exc_info=recency_data)
                recency_data = None
            if isinstance(relevance_l0_data, Exception):
                logger.error("[L2 Retriever] L0 Relevance task failed.", exc_info=relevance_l0_data)
                relevance_l0_data = None
            if isinstance(relevance_l1_data, Exception):
                logger.error("[L2 Retriever] L1 Relevance task failed.", exc_info=relevance_l1_data)
                relevance_l1_data = None

        except asyncio.TimeoutError:
            logger.error(f"[L2 Retriever] Retrieval operation timed out after {L2_RETRIEVAL_TIMEOUT} seconds.")
            return format_context_as_markdown(None, None, None, query) # Return empty result on timeout

        logger.info("[L2 Retriever] Parallel retrieval complete. Formatting results to XML...")
        return format_context_as_markdown(
            l0_recency_data=recency_data,
            l0_relevance_data=relevance_l0_data,
            l1_kg_data=relevance_l1_data,
            query=query
        )