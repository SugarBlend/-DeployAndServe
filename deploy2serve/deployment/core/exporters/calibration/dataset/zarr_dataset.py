from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from collections import deque
import torch
from typing import List, Optional, Any, Union, Tuple, Callable
from tqdm import tqdm
import zarr

from deploy2serve.deployment.core.exporters.calibration.dataset.interface import ChunkedDataset


class ZarrChunkedDataset(ChunkedDataset):
    def __init__(self, destination_folder: Union[str, Path], group_name: str) -> None:
        super().__init__()
        self.path: Path = Path(destination_folder).joinpath("data.zarr")
        self.group_name: str = group_name

        self.storage: Optional[Any] = None
        self.dataset: Optional[Any] = None

    def from_file(self, path: Union[str, Path] = None) -> None:
        if path:
            self.path = path
        self.storage = zarr.open(str(self.path), mode="r")
        self.dataset = self.storage[self.group_name]
        self.length = self.dataset.shape[0]
        self.chunk_size = self.dataset.chunks[0] if self.dataset.chunks else 128

    @property
    def filename(self) -> Path:
        return self.path

    def get_chunk(self, chunk_idx: int) -> List[torch.Tensor]:
        start = chunk_idx * self.chunk_size
        end = min(start + self.chunk_size, self.length)
        chunk = self.dataset[start: end]
        return [torch.from_numpy(item) for item in chunk]

    def get_data_shape(self) -> Tuple[int, int]:
        return self.dataset[:self.chunk_size].shape[2:]

    @staticmethod
    def to_file(
        tensor: torch.Tensor,
        path: Union[str, Path],
        group_name: str,
        chunk_size: int = 128
    ) -> None:
        array = tensor.cpu().numpy()
        storage = zarr.open(path, mode="a")

        if group_name in storage:
            del storage[group_name]

        shape = array.shape

        storage.create_dataset(
            name=group_name,
            shape=shape,
            chunks=(chunk_size, *shape[1:]),
            dtype=array.dtype,
            compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
        )[:] = array

    def create_dataset_file(
            self,
            fn: Callable[[Tuple[Any, ...]], torch.Tensor],
            files: List[Tuple[Any, ...]],
            chunk_size: int = 128,
            batch_write_size: int = 256
    ) -> None:
        sample_tensor = fn(files[0])
        if sample_tensor.ndim == 3:
            tensor_shape = sample_tensor.shape
        elif sample_tensor.ndim == 4:
            tensor_shape = sample_tensor.shape[1:]
        else:
            raise ValueError(f"Unsupported tensor shape: {sample_tensor.shape}")

        dtype = sample_tensor.cpu().numpy().dtype

        storage = zarr.open(str(self.path), mode="a")
        if self.group_name in storage:
            del storage[self.group_name]

        dataset = storage.create_dataset(
            name=self.group_name,
            shape=(0, *tensor_shape),
            chunks=(chunk_size, *tensor_shape),
            maxshape=(None, *tensor_shape),
            dtype=dtype,
            compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
        )

        buffer = deque()
        index = 0

        with ThreadPoolExecutor(max_workers=2) as executor:
            for tensor in tqdm(executor.map(fn, files), total=len(files), desc="Preprocess & write",
                               **self.progress_options):
                tensor = tensor.cpu().numpy()

                if tensor.ndim == len(tensor_shape):
                    tensor = np.expand_dims(tensor, axis=0)

                buffer.append(tensor)

                if len(buffer) >= batch_write_size:
                    batch = np.concatenate(list(buffer), axis=0)
                    dataset.resize((index + batch.shape[0], *dataset.shape[1:]))
                    dataset[index:index + batch.shape[0]] = batch
                    index += batch.shape[0]
                    buffer.clear()

            if buffer:
                batch = np.concatenate(list(buffer), axis=0)
                dataset.resize((index + batch.shape[0], *dataset.shape[1:]))
                dataset[index:index + batch.shape[0]] = batch
