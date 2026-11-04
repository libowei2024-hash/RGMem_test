# file: l0_memory/llm_interface/gemini_proxy_client.py

import json
import threading
from openai import OpenAI
from l0_memory.utils.logger import file_logger as logger
from .base import LLMBase
from .base_config import BaseLlmConfig

JSON_REPAIR_PROMPT_TEMPLATE = """
### Role
You are a precise JSON formatting tool. Your only task is to fix broken, non-compliant JSON text and make it valid.

### Task
The text below was intended to be a valid JSON object, but it contains syntax errors (e.g., extra commas, incorrect quotes, missing brackets, improper structure).
Examine it carefully, fix all errors, and output only the fully repaired JSON object that can be directly parsed by any standard JSON parser.

**DO NOT** add any explanations, comments, or Markdown code block markers in your output. Your output must start with `{{` and end with `}}`.

### Broken JSON Text
---
{broken_json}
---

### Repaired Valid JSON
"""


class GeminiProxyClient(LLMBase):
    """
    A specialized client for interacting with Gemini models via a proxy service.
    Supports specific Gemini parameters such as 'extra_body' for thinking configuration.
    """

    def __init__(self, config: BaseLlmConfig, client_name: str = "DefaultGeminiProxyClient"):
        super().__init__(config, client_name)

        if not self.config.api_key:
            raise ValueError("Gemini Proxy API Key not provided in configuration.")

        if not self.config.base_url:
            raise ValueError("Gemini Proxy Base URL not provided in configuration.")

        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=180,
        )
        logger.info(f"[{self.client_name}] Gemini Proxy Client initialized.")

    def get_completion(self, prompt: str, model_override: str = None, tools: list = None,
                       tool_choice: str | dict = None):
        """Fetches a completion with support for tool calls and Gemini-specific thinking budget."""
        thread_id = threading.get_ident()
        thinking_process = "N/A (Gemini Proxy does not provide a separate thinking process)"
        model_to_use = model_override or self.config.model

        try:
            messages = [{"role": "user", "content": prompt}]

            params = {
                "model": model_to_use,
                "messages": messages,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": self.config.max_tokens,
            }

            if tools:
                params["tools"] = tools
                if tool_choice:
                    params["tool_choice"] = tool_choice

            # Gemini-specific thinking configuration
            extra_body = {
                "google": {
                    "thinking_config": {
                        "thinking_budget": 8192
                    }
                }
            }

            response = self.client.chat.completions.create(**params, extra_body=extra_body)

            # Handle tool calls if present
            if response.choices[0].message.tool_calls:
                tool_calls_data = []
                for tool_call in response.choices[0].message.tool_calls:
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw_arguments": tool_call.function.arguments}

                    tool_calls_data.append({
                        "name": tool_call.function.name,
                        "arguments": arguments
                    })

                final_output = {"tool_calls": tool_calls_data}
                self._log_llm_call(prompt, model_to_use, thinking_process, json.dumps(final_output), response.usage)
                return final_output
            else:
                final_clean_output = response.choices[0].message.content or ""
                self._log_llm_call(prompt, model_to_use, thinking_process, final_clean_output, response.usage)
                return final_clean_output.strip()

        except Exception as e:
            logger.error(f"[{self.client_name}][Thread-{thread_id}] Exception during OpenAI-compatible API call: {e}",
                         exc_info=True)
            self._log_llm_call(prompt, model_to_use, "ERROR", f"Exception: {e}", {})
            return {} if tools else ""

    def _log_llm_call(self, prompt: str, model: str, thought: str, output: str, usage: object):
        """Logs detailed LLM call information for debugging and monitoring."""
        thread_id = threading.get_ident()
        if usage:
            input_tokens = getattr(usage, 'prompt_tokens', 0)
            output_tokens = getattr(usage, 'completion_tokens', 0)
            total_tokens = getattr(usage, 'total_tokens', 0)
        else:
            input_tokens, output_tokens, total_tokens = 0, 0, 0

        log_message = f"""
========================= LLM Call Details ({self.client_name} / Thread-{thread_id}) =========================
[Provider]
Gemini Proxy
[Model]
{model}

[Input Prompt]
----------------------------------------------------------------------
{prompt}
----------------------------------------------------------------------

[Thinking Process]
----------------------------------------------------------------------
{thought}
----------------------------------------------------------------------

[Final Clean Output]
----------------------------------------------------------------------
{output}
----------------------------------------------------------------------

[Token Usage]
Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}
====================================================================================================
"""
        logger.info(log_message)

    def get_structured_completion(self, prompt: str, model_override: str = None) -> dict:
        """Fetches a completion and attempts to parse the response as JSON."""
        raw_response = self.get_completion(prompt, model_override=model_override)

        if not isinstance(raw_response, str):
            if isinstance(raw_response, dict):
                return raw_response
            return {}

        try:
            pre_cleaned_response = raw_response.strip()
            if pre_cleaned_response.startswith("```json"):
                pre_cleaned_response = pre_cleaned_response[7:-3].strip()
            elif pre_cleaned_response.startswith("```"):
                pre_cleaned_response = pre_cleaned_response[3:-3].strip()
            return json.loads(pre_cleaned_response)
        except json.JSONDecodeError:
            logger.warning(f"[{self.client_name}] Fast JSON parsing failed. Initiating LLM repair...")
            logger.warning(f"--- Raw output to repair ---\n{raw_response}")

        try:
            repair_prompt = JSON_REPAIR_PROMPT_TEMPLATE.format(broken_json=raw_response)
            repair_model = model_override or self.config.model
            repaired_by_llm_response = self.get_completion(repair_prompt, model_override=repair_model)

            if not repaired_by_llm_response:
                logger.error(f"[{self.client_name}] LLM repairer returned empty content.")
                return {}

            parsed_json = json.loads(repaired_by_llm_response)
            logger.info(f"[{self.client_name}] LLM repair successful! JSON parsed correctly.")
            return parsed_json
        except json.JSONDecodeError as e_after_repair:
            logger.error(
                f"[{self.client_name}] Final attempt failed: Repaired content is still invalid JSON: {e_after_repair}")
            return {}
        except Exception as ex:
            logger.error(f"[{self.client_name}] Unexpected error during LLM repair process: {ex}", exc_info=True)
            return {}