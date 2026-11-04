# file: l0_memory/llm_interface/factory.py

from .base_config import BaseLlmConfig


class LlmFactory:
    """Factory class to create L0 LLM client instances."""

    provider_map = {
        "volc": (".volc_api_client", "VolcClient"),
        "openai": (".openai_api_client", "OpenAIClient"),
        "bailian": (".bailian_api_client", "BaiLianClient"),
        "gemini_proxy": (".gemini_proxy_client", "GeminiProxyClient"),
        "gemini": (".gemini_api_client", "GeminiClient"),
        # Register new providers here
    }

    @classmethod
    def create(cls, provider: str, config: dict, client_name: str):
        """
        Creates and returns an LLM client.

        Args:
            provider: The name of the LLM provider.
            config: Configuration dictionary.
            client_name: A unique name for the client instance.
        """
        provider_target = cls.provider_map.get(provider)
        if not provider_target:
            raise ValueError(f"Unknown L0 LLM provider: {provider}")

        module_name, class_name = provider_target
        from importlib import import_module
        llm_class = getattr(import_module(module_name, package=__package__), class_name)

        # Convert dictionary configuration to a config object
        llm_config = BaseLlmConfig(**config)

        # Instantiate and return the client
        return llm_class(config=llm_config, client_name=client_name)
