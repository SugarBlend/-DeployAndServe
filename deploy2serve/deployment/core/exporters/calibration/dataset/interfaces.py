from abc import ABC, abstractmethod
from typing import List, Any, Union
import torch
from pathlib import Path


class ChunkedDataset(ABC):
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

    @abstractmethod
    def get_length(self) -> int:
        pass

    @abstractmethod
    def get_chunk_size(self) -> int:
        pass

    @abstractmethod
    def get_data_shape(self) -> int:
        pass


class ChunkCache(ABC):
    @abstractmethod
    def get(self, key: int) -> Any:
        pass

    @abstractmethod
    def put(self, key: int, value: Any) -> None:
        pass
