import cv2
import numpy as np
import torch
from typing import Optional, Tuple, Dict
from ultralytics.data.augment import LetterBox

from deploy2serve.deployment.core.exporters.calibration.batcher import BaseBatcher, ExportConfig


class DetectionBatcher(BaseBatcher):
    def __init__(self, config: ExportConfig, dataset_name: str, shape: Tuple[int, int]) -> None:
        self.shape: Tuple[int, int] = shape
        self.letterbox: Optional[LetterBox] = None
        self.load_preprocess()
        super().__init__(config, dataset_name, shape)
        self.dtype = torch.float32

    def load_preprocess(self) -> None:
        self.letterbox = LetterBox(new_shape=self.shape)

    def transformation(self, image_path: str, *args, **kwargs) -> Dict[str, torch.Tensor]:
        if len(self.config.input_nodes) != 1:
            raise Exception("The 'yolo' detector model should have one input node, but more are "
                            "passed in the configuration.")

        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        preprocessed = self.letterbox(image=image)
        preprocessed = np.transpose(preprocessed, (2, 0, 1))[None]
        preprocessed = preprocessed / 255.0

        return {
            node: torch.from_numpy(preprocessed.astype(np.float32))
            for node in self.config.input_nodes
        }
