# file: l0_memory/llm_interface/base_config.py

from typing import Optional


class BaseLlmConfig:
    """Defines the base configuration and default parameters for L0 LLM clients."""

    def __init__(
            self,
            model: str,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            temperature: float = 0.3,
            top_p: float = 0.9,
            max_tokens: int = 4096
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens