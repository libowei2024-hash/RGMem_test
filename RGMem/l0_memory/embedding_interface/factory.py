# file: l0_memory/embedding_interface/factory.py

from .base_config import BaseEmbeddingConfig


class EmbeddingFactory:
    """Factory class to create embedding client instances based on the provider."""

    provider_map = {
        "local": (".local_client", "LocalEmbeddingClient"),
        "volc": (".volc_client", "VolcEmbeddingClient"),
        "gemini": (".gemini_client", "GeminiEmbeddingClient"),
        "openai": (".openai_client", "OpenAIEmbeddingClient"),
    }

    @classmethod
    def create(cls, provider: str, config_dict: dict):
        """
        Creates and returns an embedding client.

        Args:
            provider: The name of the embedding provider.
            config_dict: Dictionary containing configuration parameters.
        """
        provider_target = cls.provider_map.get(provider)
        if not provider_target:
            raise ValueError(f"Unknown Embedding provider: {provider}")

        module_name, class_name = provider_target
        from importlib import import_module
        embedder_class = getattr(import_module(module_name, package=__package__), class_name)

        embedder_config = BaseEmbeddingConfig(**config_dict)
        return embedder_class(config=embedder_config)
