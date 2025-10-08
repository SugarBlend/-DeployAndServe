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
from typing import Optional, Dict, Any

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
    pipeline: Compose
    data_preprocessor: PoseDataPreprocessor
    meta_data: Config
    def __init__(self, config: ExportConfig, dataset_name: str, batch_size: int, model_config: Config) -> None:
        self.model_config: Config = model_config
        super().__init__(config, dataset_name, batch_size)

    def transformation(self, image_path: str, bboxes: list[np.ndarray], *args, **kwargs) -> Dict[str, torch.Tensor]:  # noqa: ANN002, ANN003, ARG002
        if len(self.config.input_nodes) != 1:
            raise Exception("The 'sapiens' pose estimation model should have one input node, but more are "
                            "passed in the configuration.")

        preprocessed: list[torch.Tensor] = []
        data = {"img": cv2.imread(image_path.as_posix())}
        for bbox in bboxes:
            data["bbox_score"] = np.array([1.0])
            data["bbox"] = np.array(bbox).reshape(1, -1)
            data.update(self.meta_data)
            pose_data_sample = self.pipeline(data)
            pose_data_sample["inputs"] = [pose_data_sample["inputs"]]
            pose_data_sample["data_samples"] = [pose_data_sample["data_samples"]]
            batch_data = self.data_preprocessor(pose_data_sample, training=False)
            preprocessed.append(batch_data["inputs"])

        return {
            node: torch.concat(preprocessed, dim=0)
            for node in self.config.input_nodes
        }

    def load_preprocess(self) -> None:
        self.meta_data = Config(_get_dataset_metainfo(self.model_config)) # type: ignore[attr-defined]
        if hasattr(self.meta_data, "from_file") and isinstance(self.meta_data.from_file, (str, Path)):
            self.meta_data = Config().fromfile(f"{Path(mmpose_path).parent}/.mim/{self.meta_data.from_file}")

        DefaultScope.get_instance("mmpose", scope_name="mmpose")
        self.pipeline = Compose(self.model_config.val_pipeline)
        params = self.model_config.model.data_preprocessor.to_dict()
        params.pop("type")
        self.data_preprocessor = PoseDataPreprocessor(**params)
