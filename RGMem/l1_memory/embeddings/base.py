#l1_memory/embeddings/base.py
from abc import ABC, abstractmethod
from typing import Literal, Optional, List

from l1_memory.configs.embeddings.base import BaseEmbedderConfig


class EmbeddingBase(ABC):
    """Initialized a base embedding class

    :param config: Embedding configuration option class, defaults to None
    :type config: Optional[BaseEmbedderConfig], optional
    """

    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        if config is None:
            self.config = BaseEmbedderConfig()
        else:
            self.config = config

    @abstractmethod
    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]]):
        """
        Get the embedding for the given text.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        pass


    @abstractmethod
    def embed_batch(self, texts: List[str], memory_action: Optional[Literal["add", "search", "update"]]):
        """
        Get the embeddings for a list of texts.

        Args:
            texts (List[str]): The list of texts to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            List[list]: A list of embedding vectors.
        """
        pass
