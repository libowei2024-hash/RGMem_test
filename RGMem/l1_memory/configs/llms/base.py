#l1_memory/configs/llms/base.py
from abc import ABC
from typing import Dict, Optional, Union

import httpx

from l1_memory.configs.base import AzureConfig


class BaseLlmConfig(ABC):
    """
    Config for LLMs.
    """

    def __init__(
        self,
        model: Optional[Union[str, Dict]] = None,
        temperature: float = 0.5,
        api_key: Optional[str] = None,
        max_tokens: int = 5000,
        top_p: float = 0.95,
        top_k: int = 1,
        enable_vision: bool = False,
        vision_details: Optional[str] = "auto",
        # Openrouter specific
        models: Optional[list[str]] = None,
        route: Optional[str] = "fallback",
        openrouter_base_url: Optional[str] = None,
        # Openai specific
        openai_base_url: Optional[str] = None,
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
        # Ollama specific
        ollama_base_url: Optional[str] = None,
        # AzureOpenAI specific
        azure_kwargs: Optional[AzureConfig] = {},
        # AzureOpenAI specific
        http_client_proxies: Optional[Union[Dict, str]] = None,
        # DeepSeek specific
        deepseek_base_url: Optional[str] = None,
        # XAI specific
        xai_base_url: Optional[str] = None,
        # Sarvam specific
        sarvam_base_url: Optional[str] = "https://api.sarvam.ai/v1",
        # LM Studio specific
        lmstudio_base_url: Optional[str] = "http://localhost:1234/v1",
        lmstudio_response_format: dict = None,
        # vLLM specific
        vllm_base_url: Optional[str] = "http://localhost:8000/v1",
        # AWS Bedrock specific
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: Optional[str] = "us-west-2",
    ):


        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.enable_vision = enable_vision
        self.vision_details = vision_details

        # AzureOpenAI specific
        self.http_client = httpx.Client(proxies=http_client_proxies) if http_client_proxies else None

        # Openrouter specific
        self.models = models
        self.route = route
        self.openrouter_base_url = openrouter_base_url
        self.openai_base_url = openai_base_url
        self.site_url = site_url
        self.app_name = app_name

        # Ollama specific
        self.ollama_base_url = ollama_base_url

        # DeepSeek specific
        self.deepseek_base_url = deepseek_base_url

        # AzureOpenAI specific
        self.azure_kwargs = AzureConfig(**azure_kwargs) or {}

        # XAI specific
        self.xai_base_url = xai_base_url

        # Sarvam specific
        self.sarvam_base_url = sarvam_base_url

        # LM Studio specific
        self.lmstudio_base_url = lmstudio_base_url
        self.lmstudio_response_format = lmstudio_response_format

        # vLLM specific
        self.vllm_base_url = vllm_base_url

        # AWS Bedrock specific
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_region = aws_region
