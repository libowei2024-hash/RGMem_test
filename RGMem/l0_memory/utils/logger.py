# logger.py

import logging
import sys
import os
import time
from config import settings

LOG_DIR = os.path.join(settings.BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def setup_file_logger():
    """Configures a detailed logger that writes only to a file."""
    logger = logging.getLogger("FileLogger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Generate a unique log file name using a timestamp and process ID
        process_id = os.getpid()
        log_file_path = os.path.join(
            LOG_DIR,
            f"run_detail_{time.strftime('%Y%m%d_%H%M%S')}_pid{process_id}.log"
        )

        # Configure the file handler with a detailed format
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Log the location of the log file at startup
        logger.info(f"Detailed run log has started recording to this file.")
        # Also print the path to the console for user convenience
        print(f"[System Info] Detailed logs will be saved to: {log_file_path}")

    return logger


def setup_console_logger():
    """Configures a concise logger that outputs only to the console."""
    logger = logging.getLogger("ConsoleLogger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        # Use a simple formatter for console output
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# --- Instantiate loggers for global import ---
file_logger = setup_file_logger()
console_logger = setup_console_logger()