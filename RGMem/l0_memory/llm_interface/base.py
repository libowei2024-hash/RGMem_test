# file: l0_memory/llm_interface/base.py

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Union


class LLMBase(ABC):
    """Abstract base class for L0 LLM clients."""

    def __init__(self, config, client_name: str = "DefaultL0Client"):
        self.config = config
        self.client_name = client_name

    @abstractmethod
    def get_completion(self, prompt: str, model_override: Optional[str] = None) -> str:
        """Fetch a text completion from the LLM."""
        pass

    @abstractmethod
    def get_structured_completion(self, prompt: str, model_override: Optional[str] = None) -> Union[Dict, List]:
        """Fetch a completion and parse it into a structured format (Dict or List)."""
        pass