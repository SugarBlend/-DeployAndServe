from abc import ABC, abstractmethod
from colorama import Fore
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import torch
from tqdm import tqdm
from typing import Any, Union, Tuple, List, Callable

from deploy2serve.utils.logger import get_logger

import zarr
import torch

class LabelsGenerator(ABC):
    def __init__(self, dataset_folder: Union[str, Path]) -> None:
        self.dataset_folder: Path = Path(dataset_folder)

        self.logger = get_logger(self.__class__.__name__)

        custom_format = (
            f"{Fore.WHITE}{{desc}}: {{percentage:2.0f}}% {Fore.LIGHTGREEN_EX}|{{bar}}| "
            f"{Fore.WHITE}{{n_fmt}}/{{total_fmt}} [{Fore.LIGHTBLUE_EX}{{elapsed}}<{{remaining}} "
            f"{{rate_fmt}}]{Fore.RESET}"
        )
        self.progress_options = {
            "bar_format": custom_format, "position": 0, "leave": True, "ncols": 75, "colour": None,
        }

    @abstractmethod
    def generate_labels(self, *args, **kwargs) -> Any:
        pass

    @staticmethod
    def save_tensor_to_zarr(
            tensor: torch.Tensor,
            zarr_path: str,
            key: str = "dataset",
            chunk_size: int = 64
    ):
        # Преобразуем в numpy
        np_tensor = tensor.cpu().numpy()  # shape: (N, C, H, W)

        # Открываем Zarr-хранилище
        zarr_store = zarr.open(zarr_path, mode='a')

        # Получаем shape и dtype
        shape = np_tensor.shape
        dtype = np_tensor.dtype

        # Создаём dataset
        zarr_store.require_dataset(
            name=key,
            shape=shape,
            chunks=(chunk_size, *shape[1:]),  # например: (64, 3, 640, 640)
            dtype=dtype,
            compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2),
            overwrite=True
        )[:] = np_tensor  # Запись данных

    def create_dataset_file(
        self,
        fn: Callable[[Tuple[Any, ...]], str],
        files: List[Tuple[Any, ...]],
        group_key: str
    ) -> None:
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            tensors: list[torch.Tensor] = list(
                tqdm(executor.map(fn,files), total=len(files), desc="Preprocess frames", **self.progress_options),
            )
        tensors: torch.Tensor = torch.cat(tensors, dim=0)
        self.logger.debug(f"Try to store preprocessed dataset to hdf5 file by key: {group_key}.")
        self.save_tensor_to_zarr(tensors, str(self.dataset_folder.joinpath("data.zarr")), key="yolo")

        # with h5py.File(f"{self.calibration_folder}.h5", "a") as file:
        #     if self.dataset_key in file:
        #         self.logger.debug(f"Remove previous dataset by key: {self.dataset_key}.")
        #         del file[self.dataset_key]
        #     file.create_dataset(
        #         self.dataset_key,
        #         data=images,
        #         shape=images.shape,
        #         dtype="float32",
        #         compression="gzip",
        #     )
        self.logger.debug(f"Dataset: {group_key} successfully stored into: {self.dataset_folder.joinpath('data.zarr')} file.")