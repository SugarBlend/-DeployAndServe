import torch
from torch.utils.data import Dataset
from typing import Type

from deploy2serve.deployment.core.exporters.calibration.dataset.interfaces import ChunkedDataset, ChunkCache


class ChunkedDatasetLoader(Dataset):
    def __init__(self, dataset: ChunkedDataset, cache: Type[ChunkCache]) -> None:
        self.dataset: ChunkedDataset = dataset
        self.cache: ChunkCache = cache

        self.length: int = dataset.get_length()
        self.chunk_size: int = dataset.get_chunk_size()

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
