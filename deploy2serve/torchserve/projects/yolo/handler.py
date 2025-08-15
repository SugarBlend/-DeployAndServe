import base64
import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import ts.context
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import non_max_suppression, scale_boxes

sys.path.insert(0, str(Path(os.environ["VIRTUAL_ENV"]).parent))
from deploy2serve.utils.logger import get_logger


class YoloTorchServe(BaseHandler):
    def __init__(self) -> None:
        super(YoloTorchServe, self).__init__()
        self.model_shape = (384, 640)
        self.letterbox = LetterBox(self.model_shape)
        self.input_shape: List[Tuple[int, ...]] = []
        self.logger = get_logger("torchserve")

    @staticmethod
    def decode_image(requests: List[Dict[str, Any]]) -> List[np.ndarray]:
        images: List[np.ndarray] = []
        for request in requests:
            input_data = request.get("body") or request.get("data")
            if isinstance(input_data, str):
                image_bytes = base64.b64decode(input_data)
                image = Image.open(io.BytesIO(image_bytes))
            elif isinstance(input_data, (bytearray, bytes)):
                bytes_io = io.BytesIO(input_data)
                image = cv2.imdecode(np.frombuffer(bytes_io.read(), np.uint8), cv2.IMREAD_COLOR)
            else:
                raise ValueError("Unsupported input type")
            images.append(image)
        return images

    def initialize(self, context: ts.context.Context) -> None:
        super().initialize(context)
        self.logger.info(f"Model type:: {type(self.model)}")
        if isinstance(self.model, torch.jit._script.RecursiveScriptModule):
            self.model = self.model.half()
        else:
            raise NotImplementedError("At now available only TorchScript models for using.")

    def preprocess(self, data: List[Dict[str, Any]]) -> torch.Tensor:
        images = self.decode_image(data)
        tensors: List[torch.Tensor] = []
        for image in images:
            self.input_shape.append(tuple(image.shape[:2]))
            image_np = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            preprocessed = self.letterbox(image=image_np)
            preprocessed = np.transpose(preprocessed, (2, 0, 1))
            preprocessed = preprocessed / 255.0
            tensor = torch.from_numpy(preprocessed).to(self.device)
            if isinstance(self.model, torch.jit._script.RecursiveScriptModule):
                tensor = tensor.half()
            else:
                tensor = tensor.float()
            tensors.append(tensor)
        return torch.stack(tensors)

    def inference(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            results = self.model(data, *args, **kwargs)
        return results

    def postprocess(self, output: torch.Tensor) -> Any:
        try:
            detections = non_max_suppression(output)
            results: List[np.ndarray] = []

            for idx in range(len(detections)):
                boxes = scale_boxes(self.model_shape, detections[idx][:, :4], self.input_shape[idx]).cpu().numpy()
                scores = detections[idx][:, 4:5].cpu().numpy()
                classes = detections[idx][:, 5:].to(torch.int).cpu().numpy()
                results.append(np.concatenate([boxes, scores, classes], axis=1).tolist())
            self.input_shape.clear()
            return results
        except Exception as error:
            self.logger.error(error)
