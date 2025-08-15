import time
from importlib import import_module
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import non_max_suppression, scale_boxes

from deploy2serve.triton.core.base.inference_server import ProtocolType, TritonRemote
from deploy2serve.triton.core.base.service import parse_options
from deploy2serve.triton.core.configs import ServiceConfig


class EnsembleYoloTriton(TritonRemote):
    def __init__(self, url: str, model_name: str, protocol: ProtocolType) -> None:
        super(EnsembleYoloTriton, self).__init__(url, model_name, protocol)

    def preprocess(self, *args, **kwargs) -> Any:
        pass

    def postprocess(self, *args, **kwargs) -> Any:
        pass


class RegularYoloTriton(TritonRemote):
    def __init__(self, url: str, model_name: str, protocol: ProtocolType) -> None:
        super(RegularYoloTriton, self).__init__(url, model_name, protocol)
        self.vis_frame: Optional[np.ndarray] = None
        self.letterbox: Optional[LetterBox] = None
        self.model_shape: Optional[Tuple[int, int]] = None
        self.input_shape: Optional[Tuple[int, int]] = None

    async def initialize(self):
        await super().initialize()
        self.model_shape = tuple(map(int, self.metadata["inputs"][0]["shape"][2:]))
        self.letterbox = LetterBox(self.model_shape)

    def preprocess(self, image: np.ndarray) -> List[np.ndarray]:
        self.input_shape = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocessed = self.letterbox(image=image)
        preprocessed = np.transpose(preprocessed, (2, 0, 1))
        preprocessed = preprocessed / 255.0
        return [preprocessed[None]]

    def postprocess(self, output: np.ndarray) -> np.ndarray:
        output = torch.from_numpy(output[0]).cuda()
        detections = non_max_suppression(output)[0]
        boxes = scale_boxes(self.model_shape, detections[:, :4], self.input_shape).cpu().numpy()
        scores = detections[:, 4:5].reshape(-1, 1).cpu().numpy()
        classes = detections[:, 5:].reshape(-1, 1).to(torch.int).cpu().numpy()
        return np.concatenate([boxes, scores, classes], axis=1)


if __name__ == "__main__":
    args = parse_options()

    if "ensemble" in args.service_config:
        from deploy2serve.triton.core.clients.ensemble import Service
    elif "regular" in args.service_config:
        from deploy2serve.triton.core.clients.ensemble import Service
    else:
        raise Exception(
            "Configuration file of service must have word 'ensemble' of 'regular' in name of file "
            "for correct launch."
        )

    config = ServiceConfig.from_file(args.service_config)
    cls = getattr(import_module(config.server.module), config.server.cls)
    service = Service(cls, config.fastapi, config.triton, config.protocol)
    while service.runner.thread.is_alive():
        time.sleep(0.1)
