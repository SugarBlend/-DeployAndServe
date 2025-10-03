import shutil
from abc import ABC, abstractmethod
from gdown import download
import os
from omegaconf import OmegaConf
from importlib import import_module
from pathlib import Path
from roboflow import Roboflow
from math import floor
import torch
from torch.utils.data import DataLoader
from typing import Generator, Any, Type, List
from urllib.parse import urlparse, unquote

from deploy2serve.deployment.core.exporters.calibration.cache.lru import LRUChunkCache
from deploy2serve.deployment.core.exporters.calibration.dataset.interface import ChunkedDataset
from deploy2serve.deployment.core.exporters.calibration.loader import ChunkedDatasetLoader
from deploy2serve.deployment.models.calibration import RoboflowDataset, StandardDataset
from deploy2serve.deployment.models.export import ExportConfig
from deploy2serve.deployment.utils.uncompressor import Uncompress
from deploy2serve.utils.logger import get_logger


class BaseBatcher(ABC):
    def __init__(
        self,
        config: ExportConfig,
        dataset_name: str,
        batch_size: int
    ) -> None:
        self.config: ExportConfig = config
        self.batch_size: int = batch_size

        self.logger = get_logger(self.__class__.__name__)

        dataset_info = self.config.calibration.description
        subfolder = dataset_info.name if dataset_info else ""
        self.dataset_folder = self.config.calibration.cache_path.joinpath(f"calibration_dataset", subfolder)
        dataset = self.check_dataset_file(dataset_name)

        loader = ChunkedDatasetLoader(dataset, LRUChunkCache(max_chunks=2))
        self.dataloader = DataLoader(
            loader, batch_size=self.batch_size, num_workers=4, pin_memory=True, persistent_workers=True
        )

        if self.config.calibration.calibration_frames:
            self.total_frames = min(dataset.num_samples, self.config.calibration.calibration_frames)
        else:
            self.total_frames = dataset.num_samples
        self.total_frames = floor(self.total_frames / self.batch_size) + 1

    def check_dataset_file(self, dataset_name: str) -> ChunkedDataset:
        def regenerate_dataset() -> None:
            if self.config.calibration.description is not None:
                self._check_calibration_dataset()
            generator_info = self.config.calibration.labels_generator
            generator = getattr(import_module(generator_info.module),generator_info.class_name)(self.dataset_folder)
            labels = generator.generate_labels()
            self.load_preprocess()
            dataset.create_dataset_file(lambda args: self.transformation(*args), list(zip(*labels.values())))

        storage_info = self.config.calibration.storage
        cls: Type[ChunkedDataset] = getattr(import_module(storage_info.module), storage_info.class_name)
        dataset: ChunkedDataset = cls(self.dataset_folder, dataset_name)

        needs_regeneration = False
        if dataset.filename.exists():
            dataset.from_file()
            for node in self.config.input_nodes:
                shape = dataset.default_shapes.get(node)
                if not dataset.num_samples or shape is None:
                    self.logger.warning(f"Missing data for input node '{node}' — regenerating dataset.")
                    needs_regeneration = True
                    break

                node_shape = self.config.input_nodes[node]["shape"]
                if tuple(shape[1:]) != node_shape[1:]:
                    self.logger.warning(
                        f"Shape mismatch for node '{node}': expected {node_shape[1:]}, got {shape[1:]}."
                    )
                    needs_regeneration = True
                    break
        else:
            needs_regeneration = True

        if needs_regeneration:
            regenerate_dataset()
            dataset.from_file()

        return dataset

    def _check_calibration_dataset(self) -> None:
        dataset = self.config.calibration.description

        if isinstance(dataset, RoboflowDataset):
            images, annotations = [], []
            roboflow_config = self.dataset_folder.joinpath("data.yaml")
            if roboflow_config.exists():
                dataset_config = OmegaConf.load(roboflow_config)
                for field in ["train", "val", "test"]:
                    if hasattr(dataset_config, field):
                        folder = Path(getattr(dataset_config, field))
                        images = list(folder.glob("*"))
                        annotations = list(folder.parent.joinpath("labels").glob("*"))
                        break
        elif isinstance(dataset, StandardDataset):
            images = list(self.dataset_folder.joinpath("images").glob("*"))
            annotations = list(self.dataset_folder.joinpath("annotations").glob("*"))
        else:
            raise Exception(f"Passed unsupported type of calibration dataset: {type(self.config.calibration.dataset)}.")

        if not self.dataset_folder.exists() or not images or not annotations:
            if self.dataset_folder.exists():
                shutil.rmtree(self.dataset_folder)

            if isinstance(dataset, RoboflowDataset):
                api = Roboflow(api_key=dataset.api_key)
                project = api.workspace(dataset.workspace).project(dataset.project_id)
                project = project.version(dataset.version_number)
                project.download(dataset.model_format, self.dataset_folder.as_posix())
            elif isinstance(dataset, StandardDataset):
                archiver = Uncompress()

                for folder, source in {"images": dataset.images, "annotations": dataset.annotations}.items():
                    if source.startswith("file:/"):
                        parsed = urlparse(source)
                        output_file = Path(unquote(parsed.path.lstrip('/')))
                    elif Path(source).is_dir():
                        shutil.copytree(source, self.dataset_folder.joinpath(Path(source).name))
                        continue
                    else:
                        output_file = self.dataset_folder.joinpath(Path(source).name)
                        output_file.parent.mkdir(parents=True, exist_ok=True)
                        if not output_file.exists():
                            download(url=source, quiet=False, fuzzy=True, output=output_file.as_posix())
                    root_folder: str = archiver.uncompress(output_file, output_file.parent)
                    os.rename(output_file.parent.joinpath(root_folder), output_file.parent.joinpath(folder))
                    output_file.unlink()

    @abstractmethod
    def load_preprocess(self) -> None:
        pass

    @abstractmethod
    def transformation(self, *args, **kwargs) -> Any:
        pass

    def get_batch(self) -> Generator[List[torch.Tensor], Any, None]:
        for idx, items in enumerate(self.dataloader):
            if self.config.calibration.calibration_frames and idx > self.total_frames:
                break
            if idx in self.config.calibration.excluded_samples:
                continue

            casted_tensors: List[torch.Tensor] = []
            for node in self.config.input_nodes:
                dtype = getattr(torch, self.config.input_nodes[node]["precision"])
                casted_tensors.append(items[node].to(device=self.config.device, dtype=dtype).squeeze(axis=0))
            yield casted_tensors
