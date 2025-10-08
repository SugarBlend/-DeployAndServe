import cv2
import numpy as np
import torch
from typing import Dict
from ultralytics.data.augment import LetterBox

from deploy2serve.deployment.core.exporters.calibration.batcher import BaseBatcher


class DetectionBatcher(BaseBatcher):
    letterbox: LetterBox
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        input_node = list(self.config.input_nodes)[0]
        # FIXME: Incorrect calibration when matching types, in theory it should be float16, but the output markings
        #  are missing during calibration
        self.config.input_nodes[input_node]["precision"] = "float32"

    def load_preprocess(self) -> None:
        input_node = list(self.config.input_nodes)[0]
        bs, ch, h, w = self.config.input_nodes[input_node]["shape"]
        self.letterbox = LetterBox(new_shape=(h, w))

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
