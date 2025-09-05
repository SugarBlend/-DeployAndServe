from abc import ABC, abstractmethod
from pathlib import Path
import torch
from typing import List, Any, Union, Dict, Tuple

from deploy2serve.deployment.utils.progress_utils import get_progress_options


class ChunkedDataset(ABC):
    def __init__(self) -> None:
        self.num_samples: Dict[str, int] = {}
        self.chunk_size: Dict[str, int] = {}
        self.data_shape: Dict[str, Tuple[int, int]] = {}
        self.progress_options: Dict[str, Any] = get_progress_options()

    @property
    @abstractmethod
    def filename(self) -> Path:
        pass

    @abstractmethod
    def from_file(self, path: Union[str, Path] = None) -> None:
        pass

    @abstractmethod
    def get_chunk(self, node: str, chunk_idx: int) -> List[torch.Tensor]:
        pass

    @abstractmethod
    def create_dataset_file(self, *args, **kwargs) -> Any:
        pass
