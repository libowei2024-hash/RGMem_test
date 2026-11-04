# file: l0_memory/embedding_interface/gemini_client.py

import os
from typing import List

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError(
        "The 'google-genai' library must be installed to use Gemini Embedding. "
        "Please install it using 'pip install google-genai'."
    )

from .base import EmbeddingBase
from .base_config import BaseEmbeddingConfig
from l0_memory.utils.logger import file_logger as logger


class GeminiEmbeddingClient(EmbeddingBase):
    """
    Client for Google Gemini Embedding models, adapted for L0 layer interface specifications.
    Supports proxy-based initialization.
    """

    def __init__(self, config: BaseEmbeddingConfig):
        super().__init__(config)

        self.config.model = self.config.model or "models/text-embedding-004"
        self.config.dimensions = self.config.dimensions or 768

        api_key = self.config.api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY must be provided in the config or environment variables.")

        http_options = {}
        if self.config.base_url:
            http_options["base_url"] = self.config.base_url
            logger.info(f"[L0-GeminiEmbeddingClient] Custom base_url detected: {self.config.base_url}")

        # Initialize the GenAI client
        self.client = genai.Client(
            api_key=api_key,
            vertexai=True,
            http_options=http_options if http_options else None,
        )

        logger.info(
            f"L0 GeminiEmbeddingClient initialized. Model: {self.config.model}, Dimensions: {self.config.dimensions}"
        )

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Internal method to process batch embedding requests."""
        if not texts or not all(isinstance(t, str) for t in texts):
            return [[] for _ in texts]

        try:
            requests = []
            for text in texts:
                clean_text = text.replace("\n", " ").strip()
                if clean_text:
                    requests.append(
                        types.EmbedContentRequest(
                            model=self.config.model,
                            content=clean_text,
                            task_type="RETRIEVAL_DOCUMENT",
                            output_dimensionality=self.config.dimensions
                        )
                    )

            if not requests:
                return [[] for _ in texts]

            response = self.client.models.batch_embed_content(requests=requests)
            return [embedding['embedding'] for embedding in response['embeddings']]

        except Exception as e:
            logger.error(f"[L0-GeminiEmbeddingClient] Error calling API: {e}", exc_info=True)
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