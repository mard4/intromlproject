import logging
import os

def setup_logger(log_file_path):
    """
    Sets up the logger to log to both console and a file.

    Args:
        log_file_path (str): The file path where logs will be saved.

    Returns:
        logger (logging.Logger): Configured logger.
    """
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.DEBUG)

    # Create file handler
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
