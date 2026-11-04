# l0_memory/core/memory_processor.py

from l0_memory.memory_generation import processor
from l0_memory.utils.logger import file_logger as logger
from config import settings
from l0_memory.utils.elasticsearch_client import ElasticsearchClient
from l0_memory.embedding_interface.factory import EmbeddingFactory
from l0_memory.utils.schema import L0Memory
from pydantic import ValidationError
from l0_memory.utils.redis_publisher import RedisPublisher
from l0_memory.llm_interface.factory import LlmFactory

class MemoryProcessor:
    """
    A stateless processor responsible for executing the complete flow from a dialogue chunk to an L0 memory.
    """

    def __init__(self):
        embedding_provider = settings.EMBEDDING_PROVIDER
        embedding_config_dict = settings.EMBEDDING_CONFIGS.get(embedding_provider)
        if not embedding_config_dict:
            raise ValueError(f"Configuration for provider '{embedding_provider}' not found in EMBEDDING_CONFIGS.")
        self.embedding_client = EmbeddingFactory.create(embedding_provider, embedding_config_dict)
        self.es_client = ElasticsearchClient()
        self.redis_publisher = RedisPublisher()
        provider = settings.LLM_PROVIDER
        provider_config = settings.L0_LLM_CONFIGS.get(provider)
        if not provider_config:
            raise ValueError(f"Configuration for provider '{provider}' not found in L0_LLM_CONFIGS.")
        factory_config = {
            "model": provider_config["worker_model"],
            "api_key": provider_config.get("api_key"),
            "base_url": provider_config.get("base_url"),
            **provider_config.get("params", {})
        }
        self.llm_client = LlmFactory.create(
            provider=provider,
            config=factory_config,
            client_name=f"MemoryProcessor{provider.capitalize()}Client"
        )
        self.worker_model_name = provider_config["worker_model"]
        logger.info(f"[MemoryProcessor] Initialized, will use LLM provider: {provider}, worker model: {self.worker_model_name}")

    def process_chunk(self, dialogue_chunk: list, user_id: str, role_id: int) -> str | None:
        """
        Processes a dialogue chunk, generates an L0 memory, and returns the created_at timestamp of the last successfully processed message.
        """
        last_processed_created_at = None
        session_key = f"{user_id}_{role_id}"

        try:
            identified_topics = processor.split_topics_from_chunk(
                dialogue_chunk, self.llm_client, model_override=self.worker_model_name
            )

            if not identified_topics:
                logger.info(f"[{session_key}] No new complete topics were identified.")
                return dialogue_chunk[-1].get('created_at') if dialogue_chunk else None

            dialogue_list_for_slicing = list(dialogue_chunk)

            for topic in identified_topics:
                start_ts = topic.get('start_created_at')
                end_ts = topic.get('end_created_at')
                if not (start_ts and end_ts): continue

                try:
                    start_index = next(
                        i for i, turn in enumerate(dialogue_list_for_slicing) if turn.get('created_at') == start_ts)
                    end_index = next(
                        i for i, turn in enumerate(dialogue_list_for_slicing) if turn.get('created_at') == end_ts)
                    topic_dialogue = dialogue_list_for_slicing[start_index: end_index + 1]
                except StopIteration:
                    logger.warning(
                        f"[{session_key}] Could not find created_at boundaries {start_ts} or {end_ts} in the current batch, skipping this topic.")
                    continue

                l0_summary_dict = processor.generate_l0_summary(
                    topic_dialogue=topic_dialogue,
                    user_id=user_id,
                    role_id=role_id,
                    start_created_at=start_ts,
                    llm_client=self.llm_client,
                    model_override=self.worker_model_name
                )
                if not l0_summary_dict: continue
                try:
                    atomic_units_list = []
                    memory_id = l0_summary_dict.get("memory_id")

                    # 1. Process core_fact
                    core_fact_content = l0_summary_dict.get("core_fact", "")
                    if core_fact_content:
                        core_fact_unit = {
                            "unit_id": f"{memory_id}_core",
                            "unit_type": "core_fact",
                            "content": core_fact_content,
                            "content_vector": self.embedding_client.embed_query(core_fact_content)
                        }
                        atomic_units_list.append(core_fact_unit)

                    # 2. Loop through and process conclusions
                    for i, conclusion in enumerate(l0_summary_dict.get("conclusions", [])):
                        conclusion_content = conclusion.get("content", "")
                        if conclusion_content:
                            conclusion_unit = {
                                "unit_id": f"{memory_id}_conclusion_{i}",
                                "unit_type": "conclusion",
                                "content": conclusion_content,
                                "tag": conclusion.get("tag"),
                                "content_vector": self.embedding_client.embed_query(conclusion_content)
                            }
                            atomic_units_list.append(conclusion_unit)
                    l0_summary_dict['atomic_units'] = atomic_units_list
                    l0_memory = L0Memory(**l0_summary_dict)
                    self.es_client.index_l0_memory(l0_memory.dict())
                    self.redis_publisher.publish_memory_id(l0_memory.memory_id)
                    last_processed_created_at = end_ts
                except ValidationError as e:
                    logger.error(f"[{session_key}] L0 summary returned by LLM failed Pydantic validation: {e}")
                    logger.error(f"Original summary content: {l0_summary_dict}")
            return last_processed_created_at
        except Exception as e:
            logger.error(f"[{session_key}] An unknown error occurred while processing the dialogue chunk: {e}", exc_info=True)
            return None