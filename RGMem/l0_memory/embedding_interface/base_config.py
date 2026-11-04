# file: l0_memory/embedding_interface/base_config.py

from typing import Optional, Dict


class BaseEmbeddingConfig:
    """Defines the base configuration and default parameters for L0 Embedding clients."""

    def __init__(
            self,
            model: str,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            dimensions: Optional[int] = None,
            model_kwargs: Optional[Dict] = None,  # for local models
            encode_kwargs: Optional[Dict] = None,  # for local models
            device: str = 'auto'
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.dimensions = dimensions
        self.model_kwargs = model_kwargs or {}
        self.encode_kwargs = encode_kwargs or {}
        self.device = device