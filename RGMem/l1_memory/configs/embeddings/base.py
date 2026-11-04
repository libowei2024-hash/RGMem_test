#l1_memory/configs/embeddings/base.py
from abc import ABC
from typing import Dict, Optional, Union

import httpx

from l1_memory.configs.base import AzureConfig


class BaseEmbedderConfig(ABC):
    """
    Config for Embeddings.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_dims: Optional[int] = None,
        # Ollama specific
        ollama_base_url: Optional[str] = None,
        # Openai specific
        openai_base_url: Optional[str] = None,
        # Huggingface specific
        model_kwargs: Optional[dict] = None,
        huggingface_base_url: Optional[str] = None,
        # AzureOpenAI specific
        azure_kwargs: Optional[AzureConfig] = {},
        http_client_proxies: Optional[Union[Dict, str]] = None,
        # VertexAI specific
        vertex_credentials_json: Optional[str] = None,
        memory_add_embedding_type: Optional[str] = None,
        memory_update_embedding_type: Optional[str] = None,
        memory_search_embedding_type: Optional[str] = None,
        # Gemini specific
        output_dimensionality: Optional[str] = None,
        # LM Studio specific
        lmstudio_base_url: Optional[str] = "http://localhost:1234/v1",
        # AWS Bedrock specific
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: Optional[str] = "us-west-2",
    ):


        self.model = model
        self.api_key = api_key
        self.openai_base_url = openai_base_url
        self.embedding_dims = embedding_dims

        # AzureOpenAI specific
        self.http_client = httpx.Client(proxies=http_client_proxies) if http_client_proxies else None

        # Ollama specific
        self.ollama_base_url = ollama_base_url

        # Huggingface specific
        self.model_kwargs = model_kwargs or {}
        self.huggingface_base_url = huggingface_base_url
        # AzureOpenAI specific
        self.azure_kwargs = AzureConfig(**azure_kwargs) or {}

        # VertexAI specific
        self.vertex_credentials_json = vertex_credentials_json
        self.memory_add_embedding_type = memory_add_embedding_type
        self.memory_update_embedding_type = memory_update_embedding_type
        self.memory_search_embedding_type = memory_search_embedding_type

        # Gemini specific
        self.output_dimensionality = output_dimensionality

        # LM Studio specific
        self.lmstudio_base_url = lmstudio_base_url

        # AWS Bedrock specific
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_region = aws_region
