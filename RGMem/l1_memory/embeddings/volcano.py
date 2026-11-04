# file: l1_memory/embeddings/volcano.py

import os
import numpy as np
from typing import List, Literal, Optional

from openai import OpenAI

from l1_memory.configs.embeddings.base import BaseEmbedderConfig
from l1_memory.embeddings.base import EmbeddingBase
from l1_memory.utils import logger


class VolcanoEmbedding(EmbeddingBase):
    """
    Adapter for Volcano Engine (ByteDance) embedding models, compatible with the L1 memory framework.
    It reuses a base_url from the config if provided, otherwise defaults to the public endpoint.
    """

    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.api_key = self.config.api_key or os.environ.get("ARK_API_KEY")
        if not self.api_key:
            raise ValueError("ARK_API_KEY is required, either in config or as an environment variable.")

        self._clear_proxy_env_vars()

        # Prioritize the openai_base_url from the config (for consistency across LLM/Embedding clients),
        # falling back to the default public endpoint. Note: The BaseEmbedderConfig reuses
        # this field for this purpose.
        base_url = self.config.openai_base_url or "https://ark.cn-beijing.volces.com/api/v3"

        self.api_client = OpenAI(
            api_key=self.api_key,
            base_url=base_url
        )

        # Model name and target dimensions are read directly from the configuration object.
        self.model_name = self.config.model
        if not self.model_name:
            raise ValueError("Embedding model name is required in the config.")

        # In L1's BaseEmbedderConfig, the field is named 'embedding_dims'.
        self.target_dim = self.config.embedding_dims

        logger.info(
            f"VolcanoEmbedding initialized. Base URL: {base_url}, Model: {self.model_name}, Target Dimension: {self.target_dim}")

    def _clear_proxy_env_vars(self):
        """Removes proxy environment variables to ensure direct connection."""
        proxy_keys = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']
        for key in proxy_keys:
            if key in os.environ:
                logger.info(f"[VolcanoEmbedding] Clearing proxy environment variable: {key}")
                del os.environ[key]

    def _sliced_norm_l2(self, vec: List[float], dim: int) -> List[float]:
        """Truncates and L2-normalizes a vector."""
        sliced_vec = vec[:dim]
        norm = float(np.linalg.norm(sliced_vec))
        if norm == 0.0:
            return sliced_vec
        return [v / norm for v in sliced_vec]

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Handles batch embedding requests with client-side post-processing."""
        if not texts: return []
        try:
            response = self.api_client.embeddings.create(model=self.model_name, input=texts)
            sorted_embeddings = sorted(response.data, key=lambda e: e.index)
            full_vectors = [item.embedding for item in sorted_embeddings]
            target_dim = self.target_dim or (len(full_vectors[0]) if full_vectors else 0)
            if not target_dim: return []
            return [self._sliced_norm_l2(vec, target_dim) for vec in full_vectors]
        except Exception as e:
            logger.error(f"[VolcanoEmbedding] API call or processing failed: {e}", exc_info=True)
            return [[] for _ in texts]

    def embed(self, text: str, memory_action: Optional[Literal["add", "search", "update"]] = None) -> List[float]:
        """Embeds a single text string."""
        if not isinstance(text, str) or not text.strip(): return []
        results = self._embed_batch([text])
        return results[0] if results else []