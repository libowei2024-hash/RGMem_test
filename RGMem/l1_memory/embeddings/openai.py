# file: l1_memory/embeddings/openai.py

import os
from typing import Literal, Optional, List

from openai import OpenAI, OpenAIError
from openai import APIConnectionError, APITimeoutError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from l1_memory.configs.embeddings.base import BaseEmbedderConfig
from l1_memory.embeddings.base import EmbeddingBase
from l1_memory.utils import logger


class OpenAIEmbedding(EmbeddingBase):
    """
    Adapter for OpenAI Embedding models, compatible with the L1 memory layer framework.
    Provides standard logging and error handling.
    """

    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.config.model = self.config.model or "text-embedding-3-small"
        self.config.embedding_dims = self.config.embedding_dims or 1536

        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be provided in the config or environment variables.")

        base_url = self.config.openai_base_url or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=60)

        logger.info(
            f"OpenAIEmbedding client for L1 initialized. Model: {self.config.model}, "
            f"Dimensions: {self.config.embedding_dims}, Base URL: {base_url}"
        )

    @retry(
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError, RateLimitError)),
        stop=stop_after_attempt(5),
        wait=wait_fixed(2),
        reraise=True
    )

    def embed_batch(self, texts: List[str], memory_action: Optional[Literal["add", "search", "update"]] = None) -> List[
        List[float]]:
        """
        Generates embeddings for a list of texts using OpenAI.
        """
        if not texts or not all(isinstance(t, str) for t in texts):
            logger.warning("Input list for embedding is empty or contains non-string elements, returning empty list.")
            return [[] for _ in texts]

        try:
            # Clean texts, as the OpenAI API dislikes newlines and empty strings.
            clean_texts = [t.replace("\n", " ").strip() for t in texts]
            # Replace empty strings with a single space to prevent API errors.
            processed_texts = [t if t else " " for t in clean_texts]

            response = self.client.embeddings.create(
                input=processed_texts,
                model=self.config.model,
                dimensions=self.config.embedding_dims
            )

            # Return embeddings in their original order
            return [item.embedding for item in sorted(response.data, key=lambda e: e.index)]

        except (APIConnectionError, APITimeoutError, RateLimitError) as e:
            logger.warning(f"OpenAI Embedding connection unstable, retrying... Error: {e}")
            raise e
        except OpenAIError as e:
            logger.error(f"OpenAI Embedding API business error (non-retriable): {e}", exc_info=True)
            return [[] for _ in texts]
        except Exception as e:
            logger.error(f"An unknown error occurred with OpenAI Embedding: {e}", exc_info=True)
            return [[] for _ in texts]

    def embed(self, text: str, memory_action: Optional[Literal["add", "search", "update"]] = None) -> List[float]:
        """
        Generates an embedding for a single text by calling the batch method.
        """
        if not isinstance(text, str) or not text.strip():
            logger.warning("Input text for embedding is empty, returning an empty list.")
            return []

        results = self.embed_batch([text], memory_action=memory_action)
        return results[0] if results else []