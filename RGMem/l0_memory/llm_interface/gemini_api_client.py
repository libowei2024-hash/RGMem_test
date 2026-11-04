# file: l0_memory/llm_interface/gemini_api_client.py

import os
import json
import threading
from typing import Optional, Dict, List

try:
    import google.generativeai as genai
    from google.generativeai import types
except ImportError:
    raise ImportError(
        "The 'google-genai' library is required to use the Gemini client. "
        "Please install it using 'pip install google-genai'."
    )

from l0_memory.utils.logger import file_logger as logger
from .base import LLMBase
from .base_config import BaseLlmConfig

JSON_REPAIR_PROMPT_TEMPLATE = """
### Role
You are a precise JSON formatting tool. Your only task is to fix broken, non-compliant JSON text and make it valid.

### Task
The text below was intended to be a valid JSON object, but it contains syntax errors (e.g., extra commas, incorrect quotes, missing brackets, improper structure).
Examine it carefully, fix all errors, and then output only the fully repaired JSON object that can be directly parsed by any standard JSON parser.

**DO NOT** add any explanations, comments, or Markdown code block markers in your output. Your output must start with `{{` or `[`.

### Broken JSON Text
---
{broken_json}
---

### Repaired Valid JSON
"""


class GeminiClient(LLMBase):
    """
    A client for interacting with Google's Gemini models,
    conforming to the L0 memory layer interface.
    """

    def __init__(self, config: BaseLlmConfig, client_name: str = "DefaultGeminiClient"):
        super().__init__(config, client_name)

        if not self.config.api_key:
            raise ValueError("GOOGLE_API_KEY must be provided in the configuration.")

        http_options = {}
        if self.config.base_url:
            http_options["base_url"] = self.config.base_url
            logger.info(f"[{self.client_name}] Custom base_url detected, calling via proxy: {self.config.base_url}")

        # Initialize Google GenAI client
        self.client = genai.Client(
            api_key=self.config.api_key,
            vertexai=True,
            http_options=http_options if http_options else None,
        )

        logger.info(f"[{self.client_name}] Google Gemini client initialized. Model: {self.config.model}")

    def _reformat_messages_for_gemini(self, messages: List[Dict[str, str]]):
        """Converts standard message format to Gemini's expected format."""
        system_instruction = None
        contents = []
        for message in messages:
            if message["role"] == "system":
                system_instruction = message["content"]
            else:
                role = "model" if message["role"] == "assistant" else message["role"]
                contents.append({'role': role, 'parts': [message['content']]})
        return system_instruction, contents

    def _generate_gemini_response(
            self,
            messages: List[Dict[str, str]],
            model_override: Optional[str] = None,
            response_format: Optional[Dict] = None
    ) -> (str, object):
        """Internal method to handle the actual Gemini API call."""
        model_to_use = model_override or self.config.model
        system_instruction, contents = self._reformat_messages_for_gemini(messages)

        generation_config_dict = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_output_tokens": self.config.max_tokens,
        }

        if response_format and response_format.get("type") == "json_object":
            generation_config_dict["response_mime_type"] = "application/json"

        generation_config = types.GenerateContentConfig(**generation_config_dict)

        try:
            response = self.client.models.generate_content(
                model=model_to_use,
                contents=contents,
                generation_config=generation_config,
                system_instruction=system_instruction
            )
            return response.text, response
        except Exception as e:
            logger.error(f"[{self.client_name}] Error calling Gemini API: {e}", exc_info=True)
            return "", None

    def _log_llm_call(self, prompt: str, model: str, output: str, usage_metadata: object):
        """Logs a detailed record of the Gemini call, including token usage."""
        thread_id = threading.get_ident()

        input_tokens = 0
        output_tokens = 0
        if usage_metadata:
            input_tokens = getattr(usage_metadata, 'prompt_token_count', 0)
            output_tokens = getattr(usage_metadata, 'candidates_token_count', 0)
        total_tokens = input_tokens + output_tokens

        log_message = f"""
========================= LLM Call Details ({self.client_name} / Thread-{thread_id}) =========================
[Provider]
Google Gemini
[Model]
{model}

[Input Prompt]
----------------------------------------------------------------------
{prompt}
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

    def get_completion(self, prompt: str, model_override: Optional[str] = None) -> str:
        """Fetches a text completion from the Gemini model."""
        model_to_use = model_override or self.config.model
        messages = [{"role": "user", "content": prompt}]

        final_clean_output, response_obj = self._generate_gemini_response(messages, model_override)

        usage_metadata = getattr(response_obj, 'usage_metadata', None)
        self._log_llm_call(prompt, model_to_use, final_clean_output, usage_metadata)

        return final_clean_output.strip()

    def get_structured_completion(self, prompt: str, model_override: Optional[str] = None) -> Dict | List:
        """
        Fetches a JSON completion from the Gemini model, with a robust repair fallback.
        """
        model_to_use = model_override or self.config.model
        messages = [{"role": "user", "content": prompt}]

        # Attempt structured JSON generation
        raw_response, response_obj = self._generate_gemini_response(
            messages, model_override, response_format={"type": "json_object"}
        )

        try:
            # Pre-clean raw output
            pre_cleaned_response = raw_response.strip()
            if pre_cleaned_response.startswith("```json"):
                pre_cleaned_response = pre_cleaned_response[7:-3].strip()
            elif pre_cleaned_response.startswith("```"):
                pre_cleaned_response = pre_cleaned_response[3:-3].strip()

            parsed_json = json.loads(pre_cleaned_response)
            usage_metadata = getattr(response_obj, 'usage_metadata', None)
            self._log_llm_call(prompt, model_to_use, raw_response, usage_metadata)
            return parsed_json
        except json.JSONDecodeError:
            logger.warning(f"[{self.client_name}] Direct JSON parsing failed. Initiating LLM repair...")
            logger.warning(f"--- Raw output to repair ---\n{raw_response}")

        # Fallback repair mechanism
        try:
            repair_prompt = JSON_REPAIR_PROMPT_TEMPLATE.format(broken_json=raw_response)
            repaired_by_llm_response = self.get_completion(repair_prompt, model_override=model_to_use)

            if not repaired_by_llm_response:
                logger.error(f"[{self.client_name}] LLM repairer returned empty content.")
                return {}

            parsed_json = json.loads(repaired_by_llm_response)
            logger.info(f"[{self.client_name}] LLM repair successful!")
            return parsed_json
        except json.JSONDecodeError as e_after_repair:
            logger.error(
                f"[{self.client_name}] Final attempt failed: Repaired content is still invalid JSON: {e_after_repair}")
            return {}
        except Exception as ex:
            logger.error(f"[{self.client_name}] Unexpected error during LLM repair: {ex}", exc_info=True)
            return {}