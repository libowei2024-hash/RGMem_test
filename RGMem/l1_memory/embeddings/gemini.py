# file: l1_memory/embeddings/gemini.py

import os
from typing import Literal, Optional, List

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError(
        "The 'google-genai' library must be installed to use Gemini Embedding. "
        "Please install it using 'pip install google-genai'."
    )

from l1_memory.configs.embeddings.base import BaseEmbedderConfig
from l1_memory.embeddings.base import EmbeddingBase
from l1_memory.utils import logger  # Use L1's standard logger


class GoogleGenAIEmbedding(EmbeddingBase):
    """
    Adapter for Google Gemini Embedding models, compatible with the L1 memory layer framework.
    Supports robust, proxy-based initialization.
    """

    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        # Set default values if not provided
        self.config.model = self.config.model or "models/text-embedding-004"
        self.config.embedding_dims = self.config.embedding_dims or 768

        api_key = self.config.api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY must be provided in the config or environment variables.")

        # Initialize using genai.Client()
        http_options = {}
        # Reuse the openai_base_url field from L1 config as a generic proxy URL
        if self.config.openai_base_url:
            http_options["base_url"] = self.config.openai_base_url
            logger.info(f"[L1-GeminiEmbedding] Custom base_url detected, will call via proxy: {self.config.openai_base_url}")

        self.client = genai.Client(
            api_key=api_key,
            vertexai=True,  # Prefer Vertex AI protocol
            http_options=http_options if http_options else None,
        )

        logger.info(
            f"GeminiEmbedding client for L1 initialized. Model: {self.config.model}, Dimensions: {self.config.embedding_dims}"
        )

    def embed(self, text: str, memory_action: Optional[Literal["add", "search", "update"]] = None) -> List[float]:
        """
        Generates an embedding for the given text using Google Generative AI.
        """
        if not isinstance(text, str) or not text.strip():
            logger.warning("Input text for embedding is empty, returning an empty list.")
            return []

        try:
            # Prepare the API call
            text_to_embed = text.replace("\n", " ")
            config = types.EmbedContentConfig(output_dimensionality=self.config.embedding_dims)

            # Use the client.models.embed_content method
            response = self.client.models.embed_content(
                model=self.config.model,
                content=text_to_embed,  # Note: The parameter name is 'content'
                task_type="RETRIEVAL_DOCUMENT",  # Specify appropriate task type for knowledge graph construction
                config=config
            )

            return response["embedding"]

        except Exception as e:
            logger.error(f"Error calling Gemini Embedding API for L1: {e}", exc_info=True)
            # Return an empty list on error to maintain a consistent interface
            return []