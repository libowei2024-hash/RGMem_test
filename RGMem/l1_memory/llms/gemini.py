# file: l1_memory/llms/gemini.py

import os
from typing import Dict, List, Optional
import json

try:
    import google.generativeai as genai
    from google.generativeai import types
except ImportError:
    raise ImportError(
        "The 'google-genai' library must be installed to use the Gemini LLM. "
        "Please install it using 'pip install google-genai'."
    )

from l1_memory.configs.llms.base import BaseLlmConfig
from l1_memory.llms.base import LLMBase
from l1_memory.utils import logger
from l1_memory.memory.utils import extract_json


class GeminiLLM(LLMBase):
    """
    Adapter for Google Gemini models, making them compatible with the L1 memory layer's LLM framework.
    """

    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)

        if not self.config.model:
            self.config.model = "gemini-1.5-flash-latest"
            logger.warning(f"Gemini model not specified in config, defaulting to '{self.config.model}'.")

        api_key = self.config.api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY must be provided in the config or environment variables.")

        # Initialize using genai.Client()
        http_options = {}
        # Reuse the openai_base_url field from L1 config as a generic proxy URL
        if self.config.openai_base_url:
            http_options["base_url"] = self.config.openai_base_url
            logger.info(f"[L1-GeminiLLM] Custom base_url detected, will call via proxy: {self.config.openai_base_url}")

        self.client = genai.Client(
            api_key=api_key,
            vertexai=True,  # Prefer Vertex AI protocol
            http_options=http_options if http_options else None,
        )

        logger.info(f"GeminiLLM client for L1 initialized. Model: {self.config.model}")

    def _reformat_messages(self, messages: List[Dict[str, str]]):
        """Reformats standard messages for the Gemini API."""
        system_instruction = None
        contents = []
        for message in messages:
            if message["role"] == "system":
                system_instruction = message["content"]
            else:
                # Gemini uses 'model' for the assistant's role
                role = "model" if message["role"] == "assistant" else message["role"]
                contents.append({'role': role, 'parts': [message['content']]})
        return system_instruction, contents

    def _reformat_tools(self, tools: Optional[List[Dict]]):
        """Reformats tool definitions for the Gemini API."""
        if not tools:
            return None

        function_declarations = []
        for tool in tools:
            func = tool.get("function", {})
            parameters = func.get("parameters", {})

            function_declaration = types.FunctionDeclaration(
                name=func["name"],
                description=func.get("description", ""),
                parameters=parameters,
            )
            function_declarations.append(function_declaration)

        return [types.Tool(function_declarations=function_declarations)]

    def _parse_response(self, response, tools):
        """Parses the Gemini API response into the format expected by L1."""
        if tools:
            processed_response = {"content": None, "tool_calls": []}
            # Gemini can return both text and function calls in the same response
            for part in response.candidates[0].content.parts:
                if part.text:
                    processed_response["content"] = part.text
                if part.function_call:
                    fn = part.function_call
                    processed_response["tool_calls"].append(
                        {"name": fn.name, "arguments": dict(fn.args) if fn.args else {}}
                    )
            return processed_response
        else:
            return response.text

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        response_format: Optional[Dict] = None,
    ):
        """Generates a response using the Gemini model."""
        system_instruction, contents = self._reformat_messages(messages)
        formatted_tools = self._reformat_tools(tools)

        generation_config_dict = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_output_tokens": self.config.max_tokens,
        }

        if response_format and response_format.get("type") == "json_object":
            generation_config_dict["response_mime_type"] = "application/json"

        generation_config = types.GenerateContentConfig(**generation_config_dict)

        try:
            # Use client.models.generate_content for the API call
            response = self.client.models.generate_content(
                model=self.config.model,
                contents=contents,
                generation_config=generation_config,
                system_instruction=system_instruction,
                tools=formatted_tools,
            )

            # Log the raw response for debugging purposes
            try:
                logger.debug(f"Received raw response from Gemini API:\n{vars(response)}")
            except Exception:
                logger.debug(f"Received raw response from Gemini API (could not serialize):\n{response}")

            return self._parse_response(response, tools)

        except Exception as e:
            logger.error(f"Error calling Gemini API for L1: {e}", exc_info=True)
            # Return a failure-compatible format
            return {"content": f"Error: {e}", "tool_calls": []} if tools else f"Error: {e}"