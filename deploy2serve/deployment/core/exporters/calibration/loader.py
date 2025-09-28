import torch
from torch.utils.data import Dataset
from typing import Type, Dict, List

from deploy2serve.deployment.core.exporters.calibration.dataset.interface import ChunkedDataset
from deploy2serve.deployment.core.exporters.calibration.cache.interface import ChunkCache


class ChunkedDatasetLoader(Dataset):
    def __init__(self, dataset: Type[ChunkedDataset], cache: Type[ChunkCache]) -> None:
        self.dataset: Type[ChunkedDataset] = dataset
        self.cache: Type[ChunkCache] = cache

    def __len__(self) -> int:
        return self.dataset.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        items: Dict[str, torch.Tensor] = {}
        for node, shape in self.dataset.default_shapes.items():
            if not 0 <= idx < self.dataset.num_samples:
                raise IndexError(f"Index {idx} out of range")
            bs = shape[0]
            chunk_idx = idx // self.dataset.chunk_size[node]
            inner_idx = idx % (self.dataset.chunk_size[node])

            chunk = self.cache.get(node, chunk_idx)
            if chunk is None:
                chunk = self.dataset.get_chunk(node, chunk_idx).split(bs)
                self.cache.put(node, chunk_idx, chunk)
            items[node] = chunk[inner_idx]
        return items
