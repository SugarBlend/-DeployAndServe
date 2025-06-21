import logging
import sys
from pathlib import Path
import colorlog


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.Logger(name, level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    log_colors = {
        "DEBUG": "green",
        "INFO": "blue",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    }
    fmt = "%(asctime)s - %(name)s - %(log_color)s%(levelname)s%(reset)s - %(message)s"
    formatter = colorlog.ColoredFormatter(fmt, log_colors=log_colors)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_project_root() -> Path:
    return Path(__file__).parents[1]
