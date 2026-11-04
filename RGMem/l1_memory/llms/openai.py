# file: l1_memory/llms/openai.py

import json
import os
import warnings
from typing import Dict, List, Optional

from openai import OpenAI, OpenAIError, APIConnectionError, APITimeoutError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from l1_memory.configs.llms.base import BaseLlmConfig
from l1_memory.llms.base import LLMBase
from l1_memory.utils import logger
from l1_memory.memory.utils import extract_json  # Use L1's standard JSON extraction tool


class OpenAILLM(LLMBase):
    """
    An adapter for OpenAI GPT series models and OpenRouter, making them
    compatible with the L1 memory layer's LLM framework.
    """

    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)

        if not self.config.model:
            self.config.model = "gpt-4o-mini"  # Set a modern, cost-effective default model

        # Prioritize OpenRouter if the API key is set
        if os.environ.get("OPENROUTER_API_KEY"):
            self.is_openrouter = True
            api_key = os.environ.get("OPENROUTER_API_KEY")
            base_url = self.config.openrouter_base_url or "https://openrouter.ai/api/v1"
            logger.info(f"OpenAILLM client for L1 initialized (via OpenRouter). Base URL: {base_url}")
        else:
            # Fallback to standard OpenAI
            self.is_openrouter = False
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            base_url = self.config.openai_base_url or "https://api.openai.com/v1"
            logger.info(f"OpenAILLM client for L1 initialized (standard OpenAI). Base URL: {base_url}")

        if not api_key:
            raise ValueError(
                "Either OPENROUTER_API_KEY or OPENAI_API_KEY must be provided in config or environment variables.")

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=180)

    def _parse_response(self, response, tools):
        """
        Parses the OpenAI API response into the format expected by L1.
        Includes a fallback to extract JSON from the 'content' field if 'tool_calls' is empty.
        """
        message = response.choices[0].message

        if not tools:
            return message.content

        # Path 1: Prioritize standard tool calls
        if message.tool_calls:
            processed_response = {"content": message.content, "tool_calls": []}
            for tool_call in message.tool_calls:
                try:
                    arguments_str = tool_call.function.arguments
                    parsed_args = json.loads(extract_json(arguments_str))
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool call arguments: {arguments_str}")
                    parsed_args = {"error": "failed to parse arguments", "raw": arguments_str}

                processed_response["tool_calls"].append(
                    {"name": tool_call.function.name, "arguments": parsed_args}
                )
            return processed_response

        # Path 2 (Fallback): If tool_calls is empty, attempt to parse JSON from the 'content' field
        logger.warning("LLM did not use tool calls, attempting to extract JSON from 'content' as a fallback.")
        content_str = message.content
        if not content_str:
            return {"content": None, "tool_calls": []}

        try:
            json_from_content = extract_json(content_str)
            arguments = json.loads(json_from_content)

            # Assume only one tool was provided to reconstruct the tool_calls structure
            if tools and len(tools) == 1:
                tool_name = tools[0]["function"]["name"]
                logger.info(f"Fallback successful! Parsed arguments for tool '{tool_name}' from 'content'.")
                return {
                    "content": content_str,
                    "tool_calls": [{"name": tool_name, "arguments": arguments}]
                }
            else:
                logger.error(
                    "Fallback failed: Multiple tools were provided, cannot determine which one to assign the arguments to.")
                return {"content": content_str, "tool_calls": []}
        except (json.JSONDecodeError, IndexError) as e:
            logger.error(f"Fallback JSON extraction failed: {e}. Raw content: {content_str[:250]}...")
            return {"content": content_str, "tool_calls": []}

    @retry(
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError, RateLimitError)),
        stop=stop_after_attempt(4),
        wait=wait_fixed(3)
    )
    def generate_response(
            self,
            messages: List[Dict[str, str]],
            tools: Optional[List[Dict]] = None,
            tool_choice: str = "auto",
            response_format: Optional[Dict] = None,
    ):
        """Generates a response using either OpenAI or OpenRouter models."""
        params = {
            "model": self.config.model, "messages": messages,
            "temperature": self.config.temperature, "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        if self.is_openrouter:
            if self.config.models:
                params["models"] = self.config.models
                params.pop("model", None)
            extra_headers = {}
            if self.config.site_url and self.config.app_name:
                extra_headers["HTTP-Referer"] = self.config.site_url
                extra_headers["X-Title"] = self.config.app_name
            if extra_headers:
                params["extra_headers"] = extra_headers

        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        try:
            log_params = params.copy()
            if "extra_headers" in log_params:
                log_params["extra_headers"] = "[...omitted...]"
            logger.debug(
                f"Sending request to OpenAI/OpenRouter API:\n{json.dumps(log_params, indent=2, ensure_ascii=False)}"
            )
            response = self.client.chat.completions.create(**params)
            logger.debug(f"Received raw response from API:\n{response.model_dump_json(indent=2)}")
            return self._parse_response(response, tools)

        except (APIConnectionError, APITimeoutError, RateLimitError) as e:
            logger.warning(f"OpenAI API connection unstable, retrying... Error: {e}")
            raise e
        except OpenAIError as e:
            logger.error(f"A non-retriable business error occurred while calling OpenAI/OpenRouter API for L1: {e}",
                         exc_info=True)
            return {"content": f"Error: {e}", "tool_calls": []} if tools else f"Error: {e}"
        except Exception as e:
            logger.error(f"An unknown error occurred while calling OpenAI/OpenRouter API for L1: {e}", exc_info=True)
            return {"content": f"Error: {e}", "tool_calls": []} if tools else f"Error: {e}"
