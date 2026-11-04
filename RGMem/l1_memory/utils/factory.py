#l1_memory/utils/factory.py
import importlib
from typing import Optional

from l1_memory.configs.embeddings.base import BaseEmbedderConfig
from l1_memory.configs.llms.base import BaseLlmConfig



def load_class(class_type):
    module_path, class_name = class_type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class LlmFactory:
    provider_to_class = {
        "ollama": "l1_memory.llms.ollama.OllamaLLM",
        "openai": "l1_memory.llms.openai.OpenAILLM",
        "groq": "l1_memory.llms.groq.GroqLLM",
        "together": "l1_memory.llms.together.TogetherLLM",
        "aws_bedrock": "l1_memory.llms.aws_bedrock.AWSBedrockLLM",
        "litellm": "l1_memory.llms.litellm.LiteLLM",
        "azure_openai": "l1_memory.llms.azure_openai.AzureOpenAILLM",
        "openai_structured": "l1_memory.llms.openai_structured.OpenAIStructuredLLM",
        "anthropic": "l1_memory.llms.anthropic.AnthropicLLM",
        "azure_openai_structured": "l1_memory.llms.azure_openai_structured.AzureOpenAIStructuredLLM",
        "gemini": "l1_memory.llms.gemini.GeminiLLM",
        "deepseek": "l1_memory.llms.deepseek.DeepSeekLLM",
        "xai": "l1_memory.llms.xai.XAILLM",
        "sarvam": "l1_memory.llms.sarvam.SarvamLLM",
        "lmstudio": "l1_memory.llms.lmstudio.LMStudioLLM",
        "vllm": "l1_memory.llms.vllm.VllmLLM",
        "langchain": "l1_memory.llms.langchain.LangchainLLM",
        "volcano": "l1_memory.llms.volcano.VolcanoLLM",
        "gemini_proxy": "l1_memory.llms.gemini_proxy.L1GeminiProxyLLM",
    }

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            llm_instance = load_class(class_type)
            base_config = BaseLlmConfig(**config)
            return llm_instance(base_config)
        else:
            raise ValueError(f"Unsupported Llm provider: {provider_name}")


class EmbedderFactory:
    provider_to_class = {
        "openai": "l1_memory.embeddings.openai.OpenAIEmbedding",
        "ollama": "l1_memory.embeddings.ollama.OllamaEmbedding",
        "huggingface": "l1_memory.embeddings.huggingface.HuggingFaceEmbedding",
        "azure_openai": "l1_memory.embeddings.azure_openai.AzureOpenAIEmbedding",
        "gemini": "l1_memory.embeddings.gemini.GoogleGenAIEmbedding",
        "vertexai": "l1_memory.embeddings.vertexai.VertexAIEmbedding",
        "together": "l1_memory.embeddings.together.TogetherEmbedding",
        "lmstudio": "l1_memory.embeddings.lmstudio.LMStudioEmbedding",
        "langchain": "l1_memory.embeddings.langchain.LangchainEmbedding",
        "aws_bedrock": "l1_memory.embeddings.aws_bedrock.AWSBedrockEmbedding",
        "volcano": "l1_memory.embeddings.volcano.VolcanoEmbedding",
    }

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            embedder_instance = load_class(class_type)
            base_config = BaseEmbedderConfig(**config)
            return embedder_instance(base_config)
        else:
            raise ValueError(f"Unsupported Embedder provider: {provider_name}")



