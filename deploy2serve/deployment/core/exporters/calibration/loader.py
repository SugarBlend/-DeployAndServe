import torch
from torch.utils.data import Dataset
from typing import Type

from deploy2serve.deployment.core.exporters.calibration.dataset.interface import ChunkedDataset
from deploy2serve.deployment.core.exporters.calibration.cache.interface import ChunkCache


class ChunkedDatasetLoader(Dataset):
    def __init__(self, dataset: Type[ChunkedDataset], cache: Type[ChunkCache]) -> None:
        self.dataset: Type[ChunkedDataset] = dataset
        self.cache: Type[ChunkCache] = cache

        self.length: int = dataset.length
        self.chunk_size: int = dataset.chunk_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        if not 0 <= idx < self.length:
            raise IndexError(f"Index {idx} out of range")

        chunk_idx = idx // self.chunk_size
        inner_idx = idx % self.chunk_size

        chunk = self.cache.get(chunk_idx)
        if chunk is None:
            chunk = self.dataset.get_chunk(chunk_idx)
            self.cache.put(chunk_idx, chunk)

        return chunk[inner_idx]
