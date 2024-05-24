import logging
import os

def setup_logger(log_dir):
    """
    Set up and return a logger.

    Args:
    - log_dir (str): Directory to save the log file.

    Returns:
    - logging.Logger: Configured logger instance.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, 'training.log')
    
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)

    # Create handlers
    f_handler = logging.FileHandler(log_path)
    f_handler.setLevel(logging.INFO)

    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)

    # Create formatters and add them to handlers
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    f_format = logging.Formatter(format_str)
    c_format = logging.Formatter(format_str)
    f_handler.setFormatter(f_format)
    c_handler.setFormatter(c_format)

    # Add handlers to the logger
    if not logger.handlers:
        logger.addHandler(f_handler)
        logger.addHandler(c_handler)

    return logger
