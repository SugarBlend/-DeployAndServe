from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import torch
from typing import List, Any, Union, Tuple, Callable, Dict, Optional
from tqdm import tqdm
import h5py
import hdf5plugin

from deploy2serve.deployment.core.exporters.calibration.dataset.interface import ChunkedDataset


class HDF5ChunkedDataset(ChunkedDataset):
    def __init__(self, destination_folder: Union[str, Path], group_name: str) -> None:
        super().__init__()
        self.path: Path = Path(destination_folder).joinpath("data.h5")
        self.group_name: str = group_name

        self.storage: Optional[Any] = None
        self.dataset: Optional[Any] = None

    def from_file(self, path: Union[str, Path] = None) -> None:
        if path:
            self.path = Path(path)

        with h5py.File(self.path.as_posix(), "r") as storage:
            dataset = storage.get(self.group_name)
            if dataset:
                if dataset.attrs:
                    self.num_samples = dataset.attrs["num_samples"]
                    self.default_shapes = json.loads(dataset.attrs["dataset_info"])["nodes"]
                for name, data in dataset.items():
                    self.chunk_size[name] = data.chunks[0] if data.chunks else 32

    @property
    def filename(self) -> Path:
        return self.path

    def get_chunk(self, node: str, chunk_idx: int) -> torch.Tensor:
        bs = self.default_shapes[node][0]
        start = chunk_idx * self.chunk_size[node]
        end = min(start + self.chunk_size[node], self.num_samples)
        with h5py.File(self.path.as_posix(), "r") as storage:
            chunk = storage[self.group_name][node][bs * start: bs * end]
        return torch.from_numpy(chunk)

    def create_dataset_file(
        self,
        transform_fn: Callable[[Tuple[Any, ...]], Dict[str, torch.Tensor]],
        transform_args: List[Tuple[Any, ...]],
        chunk_size: int = 32,
        flush_threshold: int = 32
    ) -> None:
        sample_tensors: Dict[str, torch.Tensor] = transform_fn(transform_args[0])

        if isinstance(sample_tensors, (List, Tuple)):
            raise Exception("The return values of the conversion function must be of sequence type.")

        if not all(isinstance(v, torch.Tensor) for v in sample_tensors.values()):
            raise Exception("All values returned by fn must be torch.Tensors.")

        with h5py.File(self.path.as_posix(), "a") as storage:
            if self.group_name in storage:
                del storage[self.group_name]
            group = storage.create_group(self.group_name)

            arrays: Dict[str, h5py.Dataset] = {}
            for key, value in sample_tensors.items():
                tensor_shape = value.shape[1:]
                arrays[key] = group.create_dataset(
                    name=key,
                    shape=(0, *tensor_shape),
                    maxshape=(None, *tensor_shape),
                    chunks=(chunk_size, *tensor_shape),
                    dtype=value.detach().cpu().numpy().dtype,
                    **hdf5plugin.Blosc(cname="zstd", clevel=3,
                                       shuffle=hdf5plugin.Blosc.BITSHUFFLE)
                )

            tensors_buffer = {node: [] for node in sample_tensors}
            index = {node: 0 for node in sample_tensors}

            def _flush_buffer_for_key(key: str, idx: int) -> Tuple[str, int]:
                if not tensors_buffer[key]:
                    return key, 0

                batch = np.concatenate(tensors_buffer[key], axis=0)
                batch_size = batch.shape[0]
                arrays[key].resize((idx + batch_size, *arrays[key].shape[1:]))
                arrays[key][idx: idx + batch_size] = batch
                tensors_buffer[key].clear()
                return key, batch_size

            def _flush_all_buffers() -> None:
                with ThreadPoolExecutor(max_workers=min(len(tensors_buffer), 8)) as flush_executor:
                    futures = flush_executor.map(
                        lambda key: _flush_buffer_for_key(key, index[key]),
                        [key for key in tensors_buffer if tensors_buffer[key]]
                    )
                    for key, batch_size in futures:
                        index[key] += batch_size

            storage[self.group_name].attrs["num_samples"] = len(transform_args)

            with ThreadPoolExecutor(max_workers=2) as executor:
                for tensor_dict in tqdm(executor.map(transform_fn, transform_args), total=len(transform_args),
                                        desc="Preprocess & write", **self.progress_options):
                    for node, tensor in tensor_dict.items():
                        tensors_buffer[node].append(tensor.detach().cpu().numpy())
                        if len(tensors_buffer[node]) >= flush_threshold:
                            key, written = _flush_buffer_for_key(node, index[node])
                            index[key] += written

            _flush_all_buffers()

            dataset_info = {
                "nodes": {k: tuple(v.shape) for k, v in sample_tensors.items()},
                "created": datetime.now().strftime("%A, %B %d, %Y %H:%M:%S"),
            }
            storage[self.group_name].attrs["dataset_info"] = json.dumps(dataset_info)
