# file: l0_memory/embedding_interface/volc_client.py

import os
import numpy as np
from typing import List
from openai import OpenAI
from .base import EmbeddingBase
from .base_config import BaseEmbeddingConfig
from l0_memory.utils.logger import file_logger as logger


class VolcEmbeddingClient(EmbeddingBase):
    """Embedding client for Volcengine Ark models."""

    def __init__(self, config: BaseEmbeddingConfig):
        super().__init__(config)
        if not self.config.api_key:
            raise ValueError("ARK_API_KEY must be provided in the configuration.")

        # Ensure environment is clean for direct API connection
        self._clear_proxy_env_vars()

        self.api_client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url or "https://ark.cn-beijing.volces.com/api/v3"
        )

    def _clear_proxy_env_vars(self):
        """Remove proxy-related environment variables."""
        proxy_keys = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']
        for key in proxy_keys:
            if key in os.environ:
                del os.environ[key]

    def _sliced_norm_l2(self, vec: List[float], dim: int) -> List[float]:
        """Perform vector slicing and L2 normalization."""
        sliced_vec = vec[:dim]
        norm = float(np.linalg.norm(sliced_vec))
        if norm == 0.0:
            return sliced_vec
        return [v / norm for v in sliced_vec]

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch process embeddings with optional dimension slicing."""
        try:
            response = self.api_client.embeddings.create(model=self.config.model, input=texts)
            full_vectors = [item.embedding for item in sorted(response.data, key=lambda e: e.index)]

            target_dim = self.config.dimensions
            if not target_dim:
                target_dim = len(full_vectors[0]) if full_vectors else 0

            if not target_dim:
                return []

            return [self._sliced_norm_l2(vec, target_dim) for vec in full_vectors]
        except Exception as e:
            logger.error(f"[VolcEmbeddingClient] Failed to call or process Volcengine vectors: {e}", exc_info=True)
            return []

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text."""
        if not isinstance(text, str) or not text.strip():
            return []
        results = self._embed_batch([text])
        return results[0] if results else []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of document texts."""
        if not texts or not all(isinstance(t, str) for t in texts):
            return []
        return self._embed_batch(texts)