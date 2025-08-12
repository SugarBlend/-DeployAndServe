from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

from deploy2serve.utils.logger import get_logger
from deploy2serve.deployment.utils.progress_utils import get_progress_options


class LabelsGenerator(ABC):
    def __init__(self, dataset_folder: Union[str, Path]) -> None:
        self.dataset_folder: Path = Path(dataset_folder)

        self.logger = get_logger(self.__class__.__name__)
        self.progress_options = get_progress_options()

    @abstractmethod
    def generate_labels(self, *args, **kwargs) -> Any:
        pass
