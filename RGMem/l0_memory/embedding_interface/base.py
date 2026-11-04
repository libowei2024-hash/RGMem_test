# file: l0_memory/embedding_interface/base.py

from abc import ABC, abstractmethod
from typing import List


class EmbeddingBase(ABC):
    """Abstract base class for L0 Embedding clients."""

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query string."""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        pass