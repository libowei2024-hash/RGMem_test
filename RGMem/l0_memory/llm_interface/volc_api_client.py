# file: l0_memory/llm_interface/volc_api_client.py

import os
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


class VolcClient(LLMBase):
    """
    Client for interacting with Volcano Engine ARK models.
    Uses an OpenAI-compatible SDK for API calls.
    """

    def __init__(self, config: BaseLlmConfig, client_name: str = "DefaultVolcClient"):
        super().__init__(config, client_name)

        if not self.config.api_key:
            raise ValueError("ARK_API_KEY not provided in configuration.")

        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url or "https://ark.cn-beijing.volces.com/api/v3",
            timeout=1800,
        )
        logger.info(f"[{self.client_name}] Volcano Engine ARK Client initialized.")

    def get_completion(self, prompt: str, model_override: str = None) -> str:
        """
        Fetches a text completion from the model.
        Logs detailed call information, including the reasoning process, but returns only the clean answer.
        """
        thread_id = threading.get_ident()
        model_to_use = model_override or self.config.model
        if not model_to_use:
            raise ValueError("Model ID must be specified via model_override when calling VolcClient.get_completion.")

        thinking_process = "N/A"
        final_clean_output = ""
        usage_dict = {}

        try:
            messages = [{"role": "user", "content": prompt}]

            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
                extra_body={
                    "thinking": {
                        "type": "enabled",  # Use deep thinking capabilities
                    }
                },
            )

            # Extract thinking process (reasoning_content) if available
            if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[
                0].message.reasoning_content:
                thinking_process = response.choices[0].message.reasoning_content
            else:
                thinking_process = "N/A (Model did not provide thinking process)"

            # Extract the final answer
            final_clean_output = response.choices[0].message.content or ""

            # Extract token usage
            if response.usage:
                usage_dict = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                }

            self._log_llm_call(prompt, model_to_use, thinking_process, final_clean_output, usage_dict)

            return final_clean_output.strip()

        except Exception as e:
            logger.error(f"[{self.client_name}][Thread-{thread_id}] Exception during Volcano Engine API call: {e}",
                         exc_info=True)
            self._log_llm_call(prompt, model_to_use, "ERROR", f"Exception: {e}", {})
            return ""

    def _log_llm_call(self, prompt: str, model: str, thought: str, output: str, usage: dict):
        """Logs detailed LLM call information to the log file."""
        thread_id = threading.get_ident()
        input_tokens = usage.get('input_tokens', 0)
        output_tokens = usage.get('output_tokens', 0)
        total_tokens = input_tokens + output_tokens

        log_message = f"""
========================= LLM Call Details ({self.client_name} / Thread-{thread_id}) =========================
[Provider]
Volcano Engine ARK
[Model]
{model}

[Input Prompt]
----------------------------------------------------------------------
{prompt}
----------------------------------------------------------------------

[Thinking Process (Reasoning Content)]
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
        """
        Fetches a completion and parses the response as JSON.
        Includes a self-repair mechanism using an LLM fallback if parsing fails.
        """
        raw_response = self.get_completion(prompt, model_override=model_override)
        if not raw_response:
            return {}

        try:
            # Pre-clean raw output by stripping markdown block markers
            pre_cleaned_response = raw_response.strip()
            if pre_cleaned_response.startswith("```json"):
                pre_cleaned_response = pre_cleaned_response[7:-3].strip()
            elif pre_cleaned_response.startswith("```"):
                pre_cleaned_response = pre_cleaned_response[3:-3].strip()
            return json.loads(pre_cleaned_response)
        except json.JSONDecodeError:
            logger.warning(f"[{self.client_name}] Fast JSON parsing failed. Starting LLM repair...")
            logger.warning(f"--- Raw output to repair ---\n{raw_response}")

        try:
            repair_prompt = JSON_REPAIR_PROMPT_TEMPLATE.format(broken_json=raw_response)
            repair_model = model_override
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