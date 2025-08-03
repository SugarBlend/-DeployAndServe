import zarr
from pathlib import Path
import torch
from typing import List, Optional, Any, Union, Tuple

from deploy2serve.deployment.core.exporters.calibration.dataset.interfaces import ChunkedDataset


class ZarrChunkedDataset(ChunkedDataset):
    def __init__(self, destination_folder: Union[str, Path], dataset_name: str) -> None:
        self.path: Path = Path(destination_folder).joinpath("data.zarr")
        self.dataset_name: str = dataset_name

        self.storage: Optional[Any] = None
        self.dataset: Optional[Any] = None
        self.length: int = 0
        self.chunk_size: Optional[int] = None

    def from_file(self, path: Union[str, Path] = None) -> None:
        if path:
            self.path = path
        self.storage = zarr.open(str(self.path), mode="r")
        self.dataset = self.storage[self.dataset_name]
        self.length = self.dataset.shape[0]
        self.chunk_size = self.dataset.chunks[0] if self.dataset.chunks else 1024

    @property
    def filename(self) -> Path:
        return self.path

    def get_chunk(self, chunk_idx: int) -> List[torch.Tensor]:
        start = chunk_idx * self.chunk_size
        end = min(start + self.chunk_size, self.length)
        chunk = self.dataset[start: end]
        return [torch.from_numpy(item) for item in chunk]

    def get_length(self) -> int:
        return self.length

    def get_chunk_size(self) -> int:
        return self.chunk_size

    def get_data_shape(self) -> Tuple[int, int]:
        return self.dataset[:self.chunk_size].shape[2:]
