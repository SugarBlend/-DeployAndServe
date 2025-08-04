from abc import ABC, abstractmethod
from typing import List, Any, Union, Optional, Dict
import torch
from pathlib import Path

from deploy2serve.deployment.utils.progress_utils import get_progress_options


class ChunkedDataset(ABC):
    def __init__(self) -> None:
        self.length: int = 0
        self.chunk_size: Optional[int] = None
        self.progress_options: Dict[str, Any] = get_progress_options()

    @property
    @abstractmethod
    def filename(self) -> Path:
        pass

    @abstractmethod
    def from_file(self, path: Union[str, Path] = None) -> None:
        pass

    @abstractmethod
    def get_chunk(self, chunk_idx: int) -> List[torch.Tensor]:
        pass

    def get_length(self) -> int:
        return self.length

    def get_chunk_size(self) -> int:
        return self.chunk_size

    @abstractmethod
    def get_data_shape(self) -> int:
        pass

    @staticmethod
    @abstractmethod
    def to_file(*args, **kwargs) -> Any:
        pass

    @abstractmethod
    def create_dataset_file(self, *args, **kwargs) -> Any:
        pass
