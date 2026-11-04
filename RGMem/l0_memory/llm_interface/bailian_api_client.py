# l0_memory/llm_interface/bailian_api_client.py

import dashscope
from http import HTTPStatus
import json
import threading
import re
from l0_memory.utils.logger import file_logger as logger
from .base import LLMBase
from .base_config import BaseLlmConfig

JSON_REPAIR_PROMPT_TEMPLATE = """
### Role
You are a precise JSON formatting tool. Your only task is to fix broken, non-compliant JSON text and make it valid.

### Task
The text below was intended to be a valid JSON object but contains syntax errors (e.g., extra commas, incorrect quotes, missing brackets, improper structure).
Examine it carefully, fix all errors, and output only the fully repaired JSON object that can be directly parsed by any standard JSON parser.

**DO NOT** add any explanations, comments, or Markdown code block markers in your output. Your output must start with `{{` and end with `}}`.

### Broken JSON Text
---
{broken_json}
---

### Repaired Valid JSON
"""


class BaiLianClient(LLMBase):
    """
    Client for interacting with Alibaba Cloud's BaiLian (DashScope) LLM services.
    """

    def __init__(self, config: BaseLlmConfig, client_name: str = "DefaultBaiLianClient"):
        super().__init__(config, client_name)

        if not self.config.api_key:
            raise ValueError("DASHSCOPE_API_KEY not provided in configuration.")
        dashscope.api_key = self.config.api_key

        logger.info(f"[{self.client_name}] BaiLian Client initialized, using model: {self.config.model}")

    def get_completion(self, prompt: str, model_override: str = None) -> str:
        thread_id = threading.get_ident()
        model_to_use = model_override or self.config.model
        if not model_to_use:
            raise ValueError("Model must be specified when calling BaiLianClient.get_completion.")

        full_response_content = ""
        thinking_process = ""
        final_clean_output = ""
        usage_dict = {}

        try:
            # API call with configurable parameters
            response_stream = dashscope.Generation.call(
                model=model_to_use,
                prompt=prompt,
                result_format='message',
                stream=True,
                incremental_output=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
                extra_body={"enable_thinking": True}
            )

            for chunk in response_stream:
                if chunk.status_code == HTTPStatus.OK:
                    if chunk.output and chunk.output.choices:
                        full_response_content += chunk.output.choices[0].message.content
                    if chunk.usage:
                        usage_dict = chunk.usage
                else:
                    error_msg = f"API request failed: Code: {chunk.status_code}, Message: {chunk.message}"
                    logger.error(f"[{self.client_name}][Thread-{thread_id}] {error_msg}")
                    self._log_llm_call(prompt, model_to_use, thinking_process, f"ERROR: {error_msg}", usage_dict)
                    return ""

            # Extract thinking process if present
            thought_match = re.search(r"<\|im_start\|>thought([\s\S]*?)<\|im_end\|>", full_response_content, re.DOTALL)
            if thought_match:
                thinking_process = thought_match.group(1).strip()
                final_clean_output = full_response_content.replace(thought_match.group(0), "").strip()
            else:
                thinking_process = "N/A (Model did not provide thinking process)"
                final_clean_output = full_response_content.strip()

            self._log_llm_call(prompt, model_to_use, thinking_process, final_clean_output, usage_dict)
            return final_clean_output

        except Exception as e:
            logger.error(f"[{self.client_name}][Thread-{thread_id}] Exception occurred while calling BaiLian API: {e}",
                         exc_info=True)
            self._log_llm_call(prompt, model_to_use, "ERROR", f"Exception: {e}", {})
            return ""

    def _log_llm_call(self, prompt: str, model: str, thought: str, output: str, usage: dict):
        """Records details of the LLM call to the log file."""
        thread_id = threading.get_ident()
        input_tokens = usage.get('input_tokens', 0)
        output_tokens = usage.get('output_tokens', 0)
        total_tokens = input_tokens + output_tokens

        log_message = f"""
========================= LLM Call Details ({self.client_name} / Thread-{thread_id}) =========================
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
        """Fetches completion and attempts to parse it as JSON, with an LLM repair fallback."""
        raw_response = self.get_completion(prompt, model_override=model_override)
        if not raw_response:
            return {}

        try:
            pre_cleaned_response = raw_response.strip()
            if pre_cleaned_response.startswith("```json"):
                pre_cleaned_response = pre_cleaned_response[7:-3].strip()
            elif pre_cleaned_response.startswith("```"):
                pre_cleaned_response = pre_cleaned_response[3:-3].strip()
            return json.loads(pre_cleaned_response)
        except json.JSONDecodeError:
            logger.warning(f"[{self.client_name}] Fast JSON parsing failed. Starting LLM repair...")
            logger.warning(f"--- Original output to repair ---\n{raw_response}")

        try:
            repair_prompt = JSON_REPAIR_PROMPT_TEMPLATE.format(broken_json=raw_response)
            repair_model = model_override or self.config.model
            repaired_by_llm_response = self.get_completion(repair_prompt, model_override=repair_model)

            if not repaired_by_llm_response:
                logger.error(f"[{self.client_name}] LLM repair program failed to return any content.")
                return {}

            parsed_json = json.loads(repaired_by_llm_response)
            logger.info(
                f"[{self.client_name}] LLM repair program succeeded! Broken JSON repaired and parsed successfully.")
            return parsed_json
        except json.JSONDecodeError as e_after_repair:
            logger.error(f"[{self.client_name}] Final attempt: Parsing still failed after LLM repair: {e_after_repair}")
            return {}
        except Exception as ex:
            logger.error(f"[{self.client_name}] Unknown exception during LLM repair process: {ex}", exc_info=True)
            return {}