# file: l0_memory/embedding_interface/local_client.py

import os
from typing import List
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from .base import EmbeddingBase
from .base_config import BaseEmbeddingConfig


class LocalEmbeddingClient(EmbeddingBase):
    """Embedding client wrapping local HuggingFace models via LangChain."""

    def __init__(self, config: BaseEmbeddingConfig):
        super().__init__(config)

        device = self.config.device
        if device == 'auto':
            device = 'cuda' if "CUDA_VISIBLE_DEVICES" in os.environ else 'cpu'

        self.model_instance = HuggingFaceBgeEmbeddings(
            model_name=self.config.model,
            model_kwargs={'device': device, **self.config.model_kwargs},
            encode_kwargs=self.config.encode_kwargs
        )

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text."""
        if not isinstance(text, str) or not text.strip():
            return []
        return self.model_instance.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of document texts."""
        if not texts or not all(isinstance(t, str) for t in texts):
            return []
        return self.model_instance.embed_documents(texts)