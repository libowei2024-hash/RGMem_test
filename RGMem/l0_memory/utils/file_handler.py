# file_handler.py
import json
from .logger import file_logger as logger
import os
import threading

# Dictionary to store locks for each file path to ensure thread safety
_locks = {}
_lock_for_locks = threading.Lock()

def get_lock(filepath):
    """Retrieves or creates a threading lock for a specific file path."""
    with _lock_for_locks:
        if filepath not in _locks:
            _locks[filepath] = threading.Lock()
        return _locks[filepath]


def load_robust_json(filepath: str) -> list:
    """
    Attempts to load a JSON file robustly. Supports standard arrays
    or sequences of JSON objects that need to be wrapped.
    """
    if not os.path.exists(filepath):
        logger.error(f"Error: Input file {filepath} not found.")
        return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if the content is already a valid JSON array
        if content.strip().startswith('[') and content.strip().endswith(']'):
            logger.info("File appears to be a standard JSON array, parsing directly...")
            return json.loads(content)
        else:
            # Attempt to wrap content as a JSON array if it's a sequence of objects
            logger.info("File is not a standard JSON array, attempting to wrap and parse...")
            content = content.strip()
            if content.endswith(','):
                content = content[:-1]

            json_array_str = f"[{content}]"
            data = json.loads(json_array_str)
            logger.info(f"Successfully loaded {len(data)} records from {filepath}.")
            return data

    except json.JSONDecodeError as e:
        logger.error(f"Error: JSON decoding error while parsing {filepath}: {e}")
        logger.error("Please verify that the input file format is a sequence of valid JSON objects.")
        return []
    except Exception as e:
        logger.error(f"Unknown error while loading file {filepath}: {e}")
        return []


def save_json(data: dict, filepath: str):
    """Saves (overwrites) dictionary data into a formatted JSON file."""
    with get_lock(filepath):
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logger.info(f"Data successfully overwritten and saved to {filepath}")
        except Exception as e:
            logger.error(f"Error: Failed to save file to {filepath}: {e}")


def load_json(filepath: str) -> dict | list | None:
    """Loads a standard JSON file with thread-safe access."""
    with get_lock(filepath):
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Failed to load or parse JSON file {filepath}: {e}")
            return None


def append_to_jsonl(data: dict, filepath: str):
    """Appends a dictionary as a new line to a .jsonl file."""
    with get_lock(filepath):
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to append data to {filepath}: {e}")