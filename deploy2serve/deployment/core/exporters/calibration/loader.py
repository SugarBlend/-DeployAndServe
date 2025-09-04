import torch
from torch.utils.data import Dataset
from typing import Type, Dict, List

from deploy2serve.deployment.core.exporters.calibration.dataset.interface import ChunkedDataset
from deploy2serve.deployment.core.exporters.calibration.cache.interface import ChunkCache


class ChunkedDatasetLoader(Dataset):
    def __init__(self, dataset: Type[ChunkedDataset], cache: Type[ChunkCache]) -> None:
        self.dataset: Type[ChunkedDataset] = dataset
        self.cache: Type[ChunkCache] = cache

        self.num_samples: Dict[str, int] = dataset.num_samples
        self.chunk_size: Dict[str, int] = dataset.chunk_size

    def __len__(self) -> int:
        return max(self.num_samples.values())

    def __getitem__(self, idx: int) -> List[torch.Tensor]:
        items: List[torch.Tensor] = []
        for node in self.num_samples.keys():
            if not 0 <= idx < self.num_samples[node]:
                raise IndexError(f"Index {idx} out of range")

            chunk_idx = idx // self.chunk_size[node]
            inner_idx = idx % self.chunk_size[node]

            chunk = self.cache.get(node, chunk_idx)
            if chunk is None:
                chunk = self.dataset.get_chunk(node, chunk_idx)
                self.cache.put(node, chunk_idx, chunk)
            items.append(chunk[inner_idx])
        return items
