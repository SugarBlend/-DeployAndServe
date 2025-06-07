import logging
import sys
from pathlib import Path


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.Logger(name, level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt="%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    return logger


def get_project_root() -> Path:
    return Path(__file__).parents[1]
