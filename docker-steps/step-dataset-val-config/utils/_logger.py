from typing import Final
from typing import List

import logging

SEPARATOR: Final = '\n------------------------------------\n'


def setup_logger(
        name,
        file_path,
        level=logging.INFO
) -> logging.Logger:
    # setup handler
    handler = logging.FileHandler(file_path)
    handler.setFormatter(logging.Formatter('%(message)s'))
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())

    return logger


def close_loggers(
        loggers: List[logging.Logger]
) -> None:
    for logger in loggers:
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
