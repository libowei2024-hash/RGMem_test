# l0_memory/memory_generation/processor.py

from config import new_prompts as prompts
from l0_memory.data_processing.loader import format_dialogue_chunk
from l0_memory.utils.logger import file_logger as logger


def split_topics_from_chunk(dialogue_chunk: list, llm_client, model_override: str = None) -> list:
    """
    Calls the LLM to segment complete topics from a dialogue chunk.
    """
    formatted_chunk = format_dialogue_chunk(dialogue_chunk)
    prompt = prompts.TOPIC_SPLIT_PROMPT.format(dialogue_chunk=formatted_chunk)

    result = llm_client.get_structured_completion(prompt, model_override=model_override)

    if isinstance(result, list):
        logger.info(f"Topic segmentation successful, identified {len(result)} complete topics.")
        return result
    else:
        logger.warning(f"Topic segmentation did not return a list, but {type(result)}. Content: {result}")
        return []


def generate_l0_summary(
        topic_dialogue: list,
        user_id: str,
        role_id: int,
        start_created_at: str,
        llm_client,
        model_override: str = None
) -> dict:
    """
    Generates an L0 structured summary for an independent topic.
    """
    formatted_chunk = format_dialogue_chunk(topic_dialogue)
    # Extract session_id from the dialogue for logging purposes
    session_id = topic_dialogue[0].get('session_id', 'unknown_session')

    prompt = prompts.L0_SUMMARY_GENERATION_PROMPT.format(
        topic_dialogue_chunk=formatted_chunk,
        user_id=user_id,
        role_id=role_id,
        start_created_at=start_created_at
    )

    result = llm_client.get_structured_completion(prompt, model_override=model_override)

    if isinstance(result, dict) and "memory_id" in result:
        # Manually supplement user_id, role_id, and session_id into the metadata.
        # Since prompt examples might be static, we force-inject dynamic IDs here.
        result['metadata']['user_id'] = user_id
        result['metadata']['role_id'] = role_id
        result['metadata']['session_id'] = session_id

        # Regenerate memory_id with a specific structure to ensure uniqueness
        result['memory_id'] = f"{user_id}_{role_id}_{start_created_at}"

        logger.info(f"L0 summary generation successful, Memory ID: {result['memory_id']}")
        return result
    else:
        logger.error(f"L0 summary generation failed or format is incorrect. Return type: {type(result)}, Content: {result}")
        return {}