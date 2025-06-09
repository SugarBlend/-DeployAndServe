import os.path
from typing import Tuple

import cv2
import numpy as np
import torch
from mmcv.visualization.image import imshow_det_bboxes
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import non_max_suppression, scale_boxes

from deployment.core.executors.factory import ExtendExecutor
from deployment.models.export import ExportConfig
from utils.logger import get_project_root


class YoloExecutor(ExtendExecutor):
    def __init__(self, config: ExportConfig) -> None:
        super(YoloExecutor, self).__init__(config)
        self.letterbox = LetterBox(new_shape=self.config.input_shape)
        self.class_names = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocessed = torch.from_numpy(self.letterbox(image=image)).to(self.config.device)
        preprocessed = preprocessed.permute(2, 0, 1)
        preprocessed = preprocessed / 255.0
        return preprocessed[None]

    def postprocess(self, output: torch.Tensor, orig_shape) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if output.dim() == 3:
            detections = non_max_suppression(output)[0]
            boxes = scale_boxes(self.config.input_shape, detections[:, :4], orig_shape).cpu().numpy()
            scores = detections[:, 4:5].reshape(-1, 1).cpu().numpy()
            classes = detections[:, 5:].reshape(-1).to(torch.int).cpu().numpy()
        else:
            scores = output[:, 4:5].reshape(-1).cpu().numpy()
            boxes = output[:, :4][scores > 0.1]
            boxes = scale_boxes(self.config.input_shape, boxes, orig_shape).cpu().numpy()
            classes = output[:, 5:][scores > 0.1].to(torch.int32).reshape(-1).cpu().numpy()
            scores = scores[scores > 0.1].reshape(-1, 1)
        return boxes, scores, classes

    def plotter(self, backend) -> None:
        file_path = f"{get_project_root()}/deployment/resources/demo.jpg"
        if not os.path.exists(file_path):
            self.logger.warning(f"Demo file is not exist: {file_path}, skip visualization step")
            return

        image = cv2.imread(file_path)
        tensor = self.preprocess(image)
        output = self.infer(tensor, asynchronous=True)[0]
        boxes, scores, classes = self.postprocess(output, image.shape[:2])
        imshow_det_bboxes(
            image,
            np.concatenate([boxes, scores], axis=1),
            classes,
            self.class_names,
            bbox_color=(0, 233, 255),
            text_color=(0, 233, 255),
            thickness=2,
            show=True,
            win_name=backend,
        )
