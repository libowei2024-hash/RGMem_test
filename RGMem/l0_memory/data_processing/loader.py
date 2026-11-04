# l0_memory/data_processing/loader.py

from l0_memory.utils.file_handler import load_robust_json
from l0_memory.utils.logger import file_logger as logger


def load_and_prepare_chat_history(filepath: str) -> list:
    """
    Loads and sorts the chat history in ascending order based on the 'created_at' field.
    Earlier timestamps represent earlier conversations.
    """
    all_turns = load_robust_json(filepath)
    if not all_turns:
        return []
    try:
        all_turns.sort(key=lambda x: x.get('created_at', ''))
        logger.info(f"Chat history sorted in ascending order by timestamp, total turns: {len(all_turns)}")
        return all_turns
    except Exception as e:
        logger.error(f"Failed to sort chat history by timestamp: {e}", exc_info=True)

        return all_turns
def format_dialogue_chunk(chunk: list) -> str:
    """
    Formats a dialogue chunk using the new data format into a string that is easily understandable by an LLM.
    """
    formatted_lines = []
    for i, turn_data in enumerate(chunk):
        role_role = turn_data.get('role', 'unknown')
        content = turn_data.get('content', '')
        created_at = turn_data.get('created_at', 'UNKNOWN_TIME')
        session_id = turn_data.get('session_id', 'unknown_session')
        user_id = turn_data.get('user_id', 'unknown_user')
        role_id = turn_data.get('role_id', 'unknown_assistant')
        if role_role == 'user':
            role_display_name = turn_data.get('user_name', role_role)
            id_tag = f"[User: {user_id}]"
        elif role_role == 'assistant':
            role_display_name = turn_data.get('assistant_name', role_role)
            id_tag = f"[ASSISTANT: {role_id}]"
        else:
            role_display_name = 'unknown'
        formatted_lines.append(
            f"[Session: {session_id}]({role_display_name}){id_tag}[created_at: {created_at}]: {content}"
        )
    return "\n".join(formatted_lines)