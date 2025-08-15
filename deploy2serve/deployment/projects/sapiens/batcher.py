from collections.abc import Callable
import cv2
from pathlib import Path
import numpy as np
from mmengine.config.config import Config
from mmengine.dataset import Compose
from mmengine.registry import DefaultScope
from mmpose.registry import DATASETS
from mmpose import __file__ as mmpose_path
from mmpose.models.data_preprocessors import PoseDataPreprocessor
import torch
from typing import Tuple, Optional, Dict, Any

from deploy2serve.deployment.core.exporters.calibration.batcher import BaseBatcher, ExportConfig


def _get_dataset_metainfo(model_cfg: Config) -> Optional[Dict[str, Any]]:
    module_dict = DATASETS.module_dict

    for dataloader_name in [
            "test_dataloader", "val_dataloader", "train_dataloader"
    ]:
        if dataloader_name not in model_cfg:
            continue
        dataloader_cfg = model_cfg[dataloader_name]
        dataset_cfg = dataloader_cfg.dataset
        dataset_mmpose = module_dict.get(dataset_cfg.type, None)
        if dataset_mmpose is None:
            continue
        if hasattr(dataset_mmpose, "_load_metainfo") and isinstance(
                dataset_mmpose._load_metainfo, Callable):
            meta = dataset_mmpose._load_metainfo(
                dataset_cfg.get("metainfo", None))
            if meta is not None:
                return meta
        if hasattr(dataset_mmpose, "METAINFO"):
            return dataset_mmpose.METAINFO

    return None


class PoseBatcher(BaseBatcher):
    def __init__(self, config: ExportConfig, dataset_name: str, shape: Tuple[int, int], model_config: Config) -> None:
        self.model_config: Config = model_config

        self.pipeline: Optional[Compose] = None
        self.data_preprocessor: Optional[PoseDataPreprocessor] = None
        self.meta_data: Optional[Config] = None
        self.load_preprocess()
        super().__init__(config, dataset_name, shape)

    def transformation(self, image_path: str, bboxes: list[np.ndarray], *args, **kwargs) -> torch.Tensor:  # noqa: ANN002, ANN003, ARG002
        preprocessed: list[torch.Tensor] = []
        data = {"img": cv2.imread(str(image_path))}
        for bbox in bboxes:
            data["bbox_score"] = np.array([1.0])
            data["bbox"] = np.array(bbox).reshape(1, -1)
            data.update(self.meta_data)
            pose_data_sample = self.pipeline(data)
            pose_data_sample["inputs"] = [pose_data_sample["inputs"]]
            pose_data_sample["data_samples"] = [pose_data_sample["data_samples"]]
            batch_data = self.data_preprocessor(pose_data_sample, training=False)
            preprocessed.append(batch_data["inputs"])
        return torch.concat(preprocessed, dim=0)

    def load_preprocess(self) -> None:
        self.meta_data = Config(_get_dataset_metainfo(self.model_config)) # type: ignore[attr-defined]
        if hasattr(self.meta_data, "from_file"):
            self.meta_data = Config().fromfile(f"{Path(mmpose_path).parent}/.mim/{self.meta_data.from_file}")

        DefaultScope.get_instance("mmpose", scope_name="mmpose")
        self.pipeline = Compose(self.model_config.val_pipeline)
        params = self.model_config.model.data_preprocessor.to_dict()
        params.pop("type")
        self.data_preprocessor = PoseDataPreprocessor(**params)
        self.dtype = torch.float16
