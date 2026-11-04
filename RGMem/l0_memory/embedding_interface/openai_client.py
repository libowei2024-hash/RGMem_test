# file: l0_memory/embedding_interface/openai_client.py

import os
from typing import List
from openai import OpenAI, OpenAIError, APIConnectionError, APITimeoutError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from .base import EmbeddingBase
from .base_config import BaseEmbeddingConfig
from l0_memory.utils.logger import file_logger as logger


class OpenAIEmbeddingClient(EmbeddingBase):
    """
    Client for OpenAI Embedding models, adapted for L0 layer interface specifications.
    """

    def __init__(self, config: BaseEmbeddingConfig):
        super().__init__(config)

        self.config.model = self.config.model or "text-embedding-3-small"
        self.config.dimensions = self.config.dimensions or 1536

        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be provided in the config or environment variables.")

        base_url = self.config.base_url or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=60)

        logger.info(
            f"L0 OpenAIEmbeddingClient initialized. Model: {self.config.model}, "
            f"Dimensions: {self.config.dimensions}, Base URL: {base_url}"
        )

    @retry(
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError, RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_fixed(2)
    )
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Internal method to process batch embedding requests."""
        if not texts or not all(isinstance(t, str) for t in texts):
            return [[] for _ in texts]

        # Clean strings as OpenAI API does not handle newlines well
        clean_texts = [t.replace("\n", " ").strip() for t in texts]

        # Replace empty strings with a single space placeholder to prevent API errors
        processed_texts = [t if t else " " for t in clean_texts]

        try:
            response = self.client.embeddings.create(
                input=processed_texts,
                model=self.config.model,
                dimensions=self.config.dimensions
            )
            # Ensure embeddings are returned in the original order
            return [item.embedding for item in sorted(response.data, key=lambda e: e.index)]
        except OpenAIError as e:
            logger.error(f"[L0-OpenAIEmbeddingClient] OpenAIError during API call: {e}", exc_info=True)
            return [[] for _ in texts]
        except Exception as e:
            logger.error(f"[L0-OpenAIEmbeddingClient] Unknown error during API call: {e}", exc_info=True)
            return [[] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text."""
        if not isinstance(text, str) or not text.strip():
            return []
        results = self._embed_batch([text])
        return results[0] if results else []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of document texts."""
        return self._embed_batch(texts)