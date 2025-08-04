from abc import ABC, abstractmethod
from gdown import download
import os
from importlib import import_module
import numpy as np
from pathlib import Path
from roboflow import Roboflow
from math import ceil
import torch
from torch.utils.data import DataLoader
from typing import Generator, Optional, Any, Tuple, Type

from deploy2serve.deployment.core.exporters.calibration.cache.lru import LRUChunkCache
from deploy2serve.deployment.core.exporters.calibration.dataset.interface import ChunkedDataset
from deploy2serve.deployment.core.exporters.calibration.loader import ChunkedDatasetLoader
from deploy2serve.deployment.models.export import ExportConfig
from deploy2serve.deployment.models.dataset import RoboflowDataset, StandardDataset
from deploy2serve.deployment.utils.uncompressor import Uncompress


class BaseBatcher(ABC):
    def __init__(self, config: ExportConfig, dataset_name: str, shape: Tuple[int, int]) -> None:
        self.config: ExportConfig = config
        self.shape: Tuple[int, int] = shape

        self.dtype: Optional[torch.dtype] = None
        self.root = Path(self.config.tensorrt.output_file).parents[2]
        self.batch_size = max([item.get("max")[0] for item in self.config.tensorrt.specific.profile_shapes])

        self.dataset_folder = self.root.joinpath(f"calibration_dataset/{self.config.tensorrt.dataset.description.name}")
        dataset = self.check_dataset_file(dataset_name)
        loader = ChunkedDatasetLoader(dataset, LRUChunkCache(max_chunks=2))
        self.dataloader = DataLoader(
            loader, batch_size=self.batch_size, num_workers=4, pin_memory=True, persistent_workers=True
        )

        if self.config.tensorrt.dataset.calibration_frames:
            self.total_frames = min(dataset.get_length(), self.config.tensorrt.dataset.calibration_frames)
        else:
            self.total_frames = dataset.get_length()

    def check_dataset_file(self, dataset_name: str) -> ChunkedDataset:
        def regenerate_dataset() -> None:
            self._check_calibration_dataset()
            generator_info = self.config.tensorrt.dataset.labels_generator
            generator = getattr(import_module(generator_info.module), generator_info.cls)(self.dataset_folder)
            labels = generator.generate_labels()
            dataset.create_dataset_file(lambda args: self.transformation(*args), list(zip(*labels.values())))

        storage_info = self.config.tensorrt.dataset.data_storage
        cls: Type[ChunkedDataset] = getattr(import_module(storage_info.module), storage_info.cls)
        dataset: ChunkedDataset = cls(self.dataset_folder, dataset_name)

        if dataset.filename.exists():
            dataset.from_file()
            if not dataset.get_length() or dataset.get_data_shape() != self.config.input_shape:
                regenerate_dataset()
        else:
            regenerate_dataset()
            dataset.from_file()
        return dataset

    def _check_calibration_dataset(self) -> None:
        images = list(self.dataset_folder.joinpath("images").glob("*"))
        annotations = list(self.dataset_folder.joinpath("annotations").glob("*"))

        if any((not self.dataset_folder.exists(), not len(images), not len(annotations))):
            if self.dataset_folder.exists():
                for path in self.dataset_folder.iterdir():
                    path.unlink()

            dataset = self.config.tensorrt.dataset.description
            if isinstance(dataset, RoboflowDataset):
                api = Roboflow(api_key=dataset.api_key)
                project = api.workspace(dataset.workspace).project(dataset.project_id)
                project = project.version(dataset.version_number)
                project.download(dataset.model_format, self.dataset_folder)
            elif isinstance(dataset, StandardDataset):
                archiver = Uncompress()
                for folder, source in {"images": dataset.images_url, "annotations": dataset.annotations_url}.items():
                    output_file = self.dataset_folder.joinpath(Path(source).name)
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    if not output_file.exists():
                        download(url=source, quiet=False, fuzzy=True, output=str(output_file))
                    root_folder: str = archiver.uncompress(output_file, output_file.parent)
                    os.rename(output_file.parent.joinpath(root_folder), output_file.parent.joinpath(folder))
                    output_file.unlink()
            else:
                raise Exception(f"Passed unsupported type of calibration dataset: {type(self.config.tensorrt.dataset)}.")

    @abstractmethod
    def load_preprocess(self) -> None:
        pass

    @abstractmethod
    def transformation(self, *args, **kwargs) -> Any:
        pass

    def get_batch(self) -> Generator[np.ndarray, None, None]:
        for idx, batch in enumerate(self.dataloader):
            if self.config.tensorrt.dataset.calibration_frames and idx > ceil(self.total_frames / self.batch_size):
                return None
            if idx in self.config.tensorrt.dataset.exclude_frames:
                continue
            yield batch.to(self.config.device).to(self.dtype)
