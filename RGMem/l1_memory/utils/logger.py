# l1_memory/utils/logger.py

import logging
import os
from datetime import datetime

def setup_logging():
    """
    Configures the global logging system.

    This function should be called once at application startup.
    It sets up a logger that outputs to both the console and a timestamped log file.
    """
    # 1. Define the directory for storing log files
    # Use relative path to ensure it's inside 'l1_memory/logs' at the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    logs_dir = os.path.join(project_root, 'l1_memory', 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # 2. Create a unique timestamped log filename
    log_filename = f"l1_memory_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(logs_dir, log_filename)

    # 3. Get the root logger for 'l1_memory' and set the capture level to DEBUG
    # This ensures logs of all levels are captured; output levels are controlled by handlers.
    logger = logging.getLogger('l1_memory')
    logger.setLevel(logging.DEBUG)

    # 4. Prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # 5. Define log format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s'
    )

    # 6. Setup FileHandler - records all logs at DEBUG level and above
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 7. Setup StreamHandler (Console) - shows only INFO level and above to reduce clutter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 8. Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Notify that the logging system has started
    logger.info(f"Logging system initialized successfully. Logs are being recorded to: {log_filepath}")

    return logger