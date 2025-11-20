import logging
import sys
from typing import Optional

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = "app.log"):
    """
    Sets up a centralized logger for the application.

    This configures the root logger, so any logger instantiated via
    logging.getLogger(__name__) will inherit this configuration.

    Args:
        log_level: The minimum level of logs to capture (e.g., "INFO", "DEBUG").
        log_file: The file to which logs should be written. If None, logs are not saved to a file.
    """
    numeric_log_level = getattr(logging, log_level.upper(), logging.INFO)

    # Define a formatter for a consistent log format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Get the root logger and set its level
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_log_level)

    # Remove any existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # --- Add Console Handler ---
    # This handler streams logs to the console (standard output)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # --- Add File Handler (Optional) ---
    # This handler writes logs to a file
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logging.info(f"Logging configured with level {log_level} to file '{log_file}' and console.")
        except Exception as e:
            logging.error(f"Failed to configure file handler for log file '{log_file}': {e}", exc_info=True)
    else:
        logging.info(f"Logging configured with level {log_level} to console only (no log file).")

