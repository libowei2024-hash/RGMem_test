# file: l1_memory/llms/volcano.py

import os
import json
from typing import Dict, List, Optional
from openai import OpenAI

from l1_memory.llms.base import LLMBase
from l1_memory.configs.llms.base import BaseLlmConfig
from l1_memory.memory.utils import extract_json
from l1_memory.utils import logger


class VolcanoLLM(LLMBase):
    """
    An adapter class for Volcano Engine (Doubao) large language models,
    making them compatible with the l1_memory LLM framework and supporting
    the retrieval of the model's thinking process.
    """

    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)

        self.api_key = self.config.api_key or os.environ.get("ARK_API_KEY")
        if not self.api_key:
            raise ValueError("ARK_API_KEY is required, either in config or as an environment variable.")

        base_url = self.config.openai_base_url or "https://ark.cn-beijing.volces.com/api/v3"
        logger.info(f"Initializing VolcanoLLM client. Base URL: {base_url}")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=1800
        )

        if not self.config.model:
            # Set a default model if not provided in the configuration.
            self.config.model = "doubao-seed-1-6-thinking-250715"

        logger.info(f"VolcanoLLM client initialized. Default model: {self.config.model}")

    def _parse_response(self, response, tools):
        """
        Parses the response from the API.
        Formats tool call outputs if tools were used, and logs the 'reasoning_content' field.
        """
        # Log the thinking process if available
        if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
            thought_process = response.choices[0].message.reasoning_content
            logger.info("--- LLM Thought Process (from reasoning_content) ---")
            logger.info(f"\n{thought_process}")
            logger.info("------------------------------------------------------")

        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    arguments_str = tool_call.function.arguments
                    try:
                        parsed_args = json.loads(extract_json(arguments_str))
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool arguments: {arguments_str}")
                        parsed_args = {"error": "failed to parse arguments", "raw": arguments_str}

                    processed_response["tool_calls"].append(
                        {"name": tool_call.function.name, "arguments": parsed_args}
                    )
            return processed_response
        else:
            return response.choices[0].message.content

    def generate_response(
            self,
            messages: List[Dict[str, str]],
            tools: Optional[List[Dict]] = None,
            tool_choice: str = "auto",
            **kwargs
    ):
        """
        Generates a response using the Volcano Engine model and requests the thinking process.
        """
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

        # Add extra_body to enable the thinking process
        extra_body_params = {
            "thinking": {
                "type": "enabled",
            }
        }

        # Log parameters being sent (excluding sensitive info)
        log_params = params.copy()
        log_params.update(extra_body_params)
        logger.debug(
            f"Sending request to Volcano API with params:\n{json.dumps(log_params, indent=2, ensure_ascii=False)}")

        # Make the API call, passing extra_body
        response = self.client.chat.completions.create(**params, extra_body=extra_body_params)

        # Log the raw response
        try:
            raw_response_str = response.model_dump_json(indent=2)
            logger.debug(f"Received raw response from Volcano API:\n{raw_response_str}")
        except Exception:
            logger.debug(f"Received raw response from Volcano API (could not serialize):\n{response}")

        # The parsing function will handle logging the thought process
        return self._parse_response(response, tools)