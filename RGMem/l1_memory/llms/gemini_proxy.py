# file: l1_memory/llms/gemini_proxy.py

import os
import json
from typing import Dict, List, Optional
from openai import OpenAI

from l1_memory.llms.base import LLMBase
from l1_memory.configs.llms.base import BaseLlmConfig
from l1_memory.memory.utils import extract_json
from l1_memory.utils import logger


class L1GeminiProxyLLM(LLMBase):
    """
    An adapter class for interacting with Gemini models via a proxy service,
    specifically designed for the L1 memory layer.
    """

    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)

        self.api_key = self.config.api_key
        if not self.api_key:
            raise ValueError("Gemini Proxy API Key is required for L1.")

        # The base_url will be configured for Gemini in settings.py,
        # and this class will use it.
        base_url = self.config.openai_base_url
        if not base_url:
            raise ValueError("Gemini Proxy Base URL is required for L1.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=1800
        )
        logger.info(f"Initializing L1GeminiProxyLLM client. Base URL: {base_url}")

    def _parse_response(self, response, tools):
        """Parses the response from the API."""
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    arguments_str = tool_call.function.arguments
                    try:
                        # Ensure usage of l1_memory's utility function for extraction
                        parsed_args = json.loads(extract_json(arguments_str))
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool arguments: {arguments_str}")
                        parsed_args = {"error": "failed to parse arguments", "raw": arguments_str}
                    processed_response["tool_calls"].append({
                        "name": tool_call.function.name,
                        "arguments": parsed_args
                    })
            return processed_response
        else:
            return response.choices[0].message.content

    def generate_response(
            self,
            messages: List[Dict[str, str]],
            tools: Optional[List[Dict]] = None,
            tool_choice: str = "auto",
            response_format: Optional[Dict] = None,
            **kwargs
    ):
        """Generates a response using the Gemini Proxy model."""
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        if response_format:
            params["response_format"] = response_format

        # Add Gemini-specific extra_body for proxy compatibility
        extra_body_params = {
            "google": {"thinking_config": {"thinking_budget": 0}}
        }

        response = self.client.chat.completions.create(**params, extra_body=extra_body_params)

        return self._parse_response(response, tools)