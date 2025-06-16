import base64
import cv2
import sys
import io
import os
import numpy as np
import ts.context
from PIL import Image
from pathlib import Path
import torch
from typing import List, Dict, Any, Optional, Tuple
from ts.torch_handler.base_handler import BaseHandler
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import non_max_suppression, scale_boxes

sys.path.insert(0, str(Path(os.environ["VIRTUAL_ENV"]).parent))
from utils.logger import get_logger


class YoloTorchServe(BaseHandler):
    def __init__(self) -> None:
        super(YoloTorchServe, self).__init__()
        self.model_shape = (384, 640)
        self.letterbox = LetterBox(self.model_shape)
        self.input_shape: Optional[Tuple[int, int]] = None
        self.logger = get_logger("torchserve")

    @staticmethod
    def decode_image(data: List[Dict[str, Any]]) -> np.ndarray:
        input_data = data[0].get("body") or data[0].get("data")

        if isinstance(input_data, str):
            image_bytes = base64.b64decode(input_data)
            image = Image.open(io.BytesIO(image_bytes))
        elif isinstance(input_data, (bytearray, bytes)):
            bytes_io = io.BytesIO(input_data)
            image = cv2.imdecode(np.frombuffer(bytes_io.read(), np.uint8), cv2.IMREAD_COLOR)
        else:
            raise ValueError("Unsupported input type")

        return np.array(image)

    def initialize(self, context: ts.context.Context) -> None:
        super().initialize(context)
        self.logger.info(f"Model type:: {type(self.model)}")
        if isinstance(self.model, torch.jit._script.RecursiveScriptModule):
            self.model = self.model.half()
        else:
            raise NotImplementedError("At now available only TorchScript models for using.")

    def preprocess(self, data: List[Dict[str, Any]]) -> torch.Tensor:
        image = self.decode_image(data)
        self.input_shape = image.shape[:2]
        image_np = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        preprocessed = self.letterbox(image=image_np)
        preprocessed = np.transpose(preprocessed, (2, 0, 1))
        preprocessed = preprocessed / 255.0
        tensor = torch.from_numpy(preprocessed[None]).to(self.device)
        return tensor.half() if isinstance(self.model, torch.jit._script.RecursiveScriptModule) else tensor.float()

    def inference(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            marshalled_data = data.to(self.device)
            results = self.model(marshalled_data, *args, **kwargs)
        return results

    def postprocess(self, output: torch.Tensor) -> List[Dict[str, Any]]:
        try:
            self.logger.info(f"Output tensor shape before NMS: {output.shape}")
            self.logger.info(f"Output tensor dtype: {output.dtype}")

            detections = non_max_suppression(output)

            if not detections or len(detections[0]) == 0:
                self.logger.warning("No detections found after NMS")
                return [{"predictions": []}]

            results: List[Dict[str, List[Dict[str, Any]]]] = []
            for detection in detections:
                if detection.shape[1] < 6:
                    raise ValueError(f"Unexpected detection tensor shape: {detections.shape}")

                boxes = scale_boxes(self.model_shape, detection[:, :4], self.input_shape).cpu().numpy().tolist()
                scores = detection[:, 4].cpu().numpy().tolist()
                classes = detection[:, 5].to(torch.int).cpu().numpy().tolist()

                result: List[Dict[str, Any]] = []
                for box, score, cls in zip(boxes, scores, classes):
                    result.append({
                        "bbox": box,
                        "score": float(score),
                        "class": int(cls)
                    })
                results.append({"predictions": result})

            return results

        except Exception as error:
            self.logger.error(error)
            return [{"error": error}]
