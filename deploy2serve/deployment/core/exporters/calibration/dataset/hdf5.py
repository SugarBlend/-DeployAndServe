import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from typing import List, Any, Union, Tuple, Callable
from tqdm import tqdm
import h5py

from deploy2serve.deployment.core.exporters.calibration.dataset.interface import ChunkedDataset


class HDF5ChunkedDataset(ChunkedDataset):
    def __init__(self, destination_folder: Union[str, Path], group_name: str) -> None:
        super().__init__()
        self.path: Path = Path(destination_folder).joinpath("data.h5")
        self.group_name: str = group_name

    def from_file(self, path: Union[str, Path] = None) -> None:
        if path:
            self.path = Path(path)
        with h5py.File(str(self.path), "r") as storage:
            dataset = storage[self.group_name]
            self.length = dataset.shape[0]
            self.chunk_size = dataset.chunks[0] if dataset.chunks else 1024
            self.data_shape = dataset[:self.chunk_size].shape[2:]

    @property
    def filename(self) -> Path:
        return self.path

    def get_chunk(self, chunk_idx: int) -> List[torch.Tensor]:
        start = chunk_idx * self.chunk_size
        end = min(start + self.chunk_size, self.length)
        with h5py.File(str(self.path), "r") as storage:
            chunk = storage[self.group_name][start: end]
        return [torch.from_numpy(item) for item in chunk]

    @staticmethod
    def to_file(
        tensor: torch.Tensor,
        path: Union[str, Path],
        group_name: str,
        chunk_size: int = 1024
    ) -> None:
        array = tensor.cpu().numpy()
        with h5py.File(str(path), "a") as storage:
            if group_name in storage:
                del storage[group_name]

            shape = array.shape
            maxshape = (None,) + shape[1:]

            storage.create_dataset(
                name=group_name,
                shape=shape,
                maxshape=maxshape,
                chunks=(chunk_size,) + shape[1:],
                dtype=array.dtype,
                compression="gzip",
                compression_opts=4
            )[:] = array

    def create_dataset_file(
        self,
        fn: Callable[[Tuple[Any, ...]], torch.Tensor],
        files: List[Tuple[Any, ...]],
        chunk_size: int = 128,
        batch_write_size: int = 1024
    ) -> None:
        sample_tensor = fn(files[0])

        if sample_tensor.ndim == 3:
            tensor_shape = sample_tensor.shape
        elif sample_tensor.ndim == 4:
            tensor_shape = sample_tensor.shape[1:]
        else:
            raise ValueError(f"Unsupported tensor shape: {sample_tensor.shape}")

        dtype = sample_tensor.cpu().numpy().dtype
        total_len = len(files)
        buffer: List[np.ndarray] = []
        index = 0

        with h5py.File(str(self.path), "a") as storage:
            if self.group_name in storage:
                del storage[self.group_name]

            dataset = storage.create_dataset(
                name=self.group_name,
                shape=(total_len, *tensor_shape),
                maxshape=(total_len, *tensor_shape),
                chunks=(chunk_size, *tensor_shape),
                dtype=dtype,
                compression="gzip",
                compression_opts=4
            )

            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                for tensor in tqdm(executor.map(fn, files), total=total_len, desc="Preprocess & write",
                                   **self.progress_options):
                    tensor = tensor.cpu().numpy()
                    if tensor.ndim == len(tensor_shape):
                        tensor = np.expand_dims(tensor, axis=0)
                    buffer.append(tensor)

                    if len(buffer) * tensor.shape[0] >= batch_write_size:
                        batch = np.concatenate(buffer, axis=0)
                        batch_size = batch.shape[0]
                        dataset[index:index + batch_size] = batch
                        index += batch_size
                        buffer.clear()

                if buffer:
                    batch = np.concatenate(buffer, axis=0)
                    batch_size = batch.shape[0]
                    dataset[index:index + batch_size] = batch
