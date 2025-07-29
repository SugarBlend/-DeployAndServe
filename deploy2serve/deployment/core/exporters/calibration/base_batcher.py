from abc import ABC, abstractmethod
from typing import Generator

import numpy as np


class BaseBatcher(ABC):
    def __init__(self) -> None:
        pass

    def __parse_labels(self) -> None:
        pass

    def __check_calibration_dataset(self) -> None:
        pass

    @abstractmethod
    def load_preprocess(self) -> None:
        pass

    @abstractmethod
    def get_batch(self) -> Generator[np.ndarray, None, None]:
        pass
