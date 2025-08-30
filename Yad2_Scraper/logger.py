import logging as py_logging
from logging.handlers import RotatingFileHandler
from typing import Optional

def setup_logger(name: Optional[str] = 'scraper') -> py_logging.Logger:
    logger = py_logging.getLogger(name)
    logger.setLevel(py_logging.DEBUG)

    formatter = py_logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        file_handler = RotatingFileHandler('scraper.log', maxBytes=5*1024*1024, backupCount=3)
        file_handler.setLevel(py_logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not any(isinstance(h, py_logging.StreamHandler) for h in logger.handlers):
        console_handler = py_logging.StreamHandler()
        console_handler.setLevel(py_logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.propagate = False
    return logger

logger = setup_logger()
