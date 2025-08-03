import urllib.request
from typing import List, Tuple

import cv2
import numpy as np
import torch
from mmcv.visualization.image import imshow_det_bboxes
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import non_max_suppression, scale_boxes

from deploy2serve.deployment.core.executors.factory import ExtendExecutor
from deploy2serve.deployment.models.export import ExportConfig
from deploy2serve.deployment.models.export.common import Plugin
from deploy2serve.utils.logger import get_project_root


class YoloExecutor(ExtendExecutor):
    def __init__(self, config: ExportConfig) -> None:
        super(YoloExecutor, self).__init__(config)
        self.letterbox = LetterBox(new_shape=self.config.input_shape)
        self.class_names = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush",
        ]

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocessed = torch.from_numpy(self.letterbox(image=image)).to(self.config.device)
        preprocessed = preprocessed.permute(2, 0, 1)
        preprocessed = preprocessed / 255.0
        if self.config.enable_mixed_precision:
            preprocessed = preprocessed.half()
        else:
            preprocessed = preprocessed.float()
        return preprocessed[None]

    def postprocess(
        self, output: torch.Tensor, orig_shape
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        available_plugins = ["efficient_nms", "batched_nms"]

        plugin: List[Plugin] = []
        backend_conf = getattr(self.config, self.backend)
        if hasattr(backend_conf, "plugins"):
            plugins = backend_conf.plugins
        else:
            plugins = []

        for search_plugin in available_plugins:
            plugin = next((plugin for plugin in plugins if plugin.name == search_plugin), [])
            if plugin:
                break

        boxes: List[np.ndarray] = []
        classes: List[np.ndarray] = []
        scores: List[np.ndarray] = []
        if plugin:
            output = output[0]
            for idx in range(output.shape[0]):
                confidences = output[idx, :, 4:5].reshape(-1).cpu().numpy()
                detections = output[idx, :, :4][confidences > 0.1]
                boxes.append(scale_boxes(self.config.input_shape, detections, orig_shape).cpu().numpy())
                classes.append(output[idx, :, 5:][confidences > 0.1].to(torch.int32).reshape(-1).cpu().numpy())
                scores.append(confidences[confidences > 0.1].reshape(-1, 1))
        else:
            if len(output) > 1:
                return [], [], []
            else:
                for idx in range(len(output)):
                    detections = non_max_suppression(output[idx])[0]
                    boxes.append(scale_boxes(self.config.input_shape, detections[:, :4], orig_shape).cpu().numpy())
                    scores.append(detections[:, 4:5].reshape(-1, 1).cpu().numpy())
                    classes.append(detections[:, 5:].reshape(-1).to(torch.int).cpu().numpy())

        return boxes, scores, classes

    def plotter(self) -> None:
        file_path = get_project_root().joinpath("resources/demo.jpg")
        if not file_path.exists():
            self.logger.warning(f"Demo file is not exist: {file_path}, skip visualization step")
            image_url = "https://ultralytics.com/images/zidane.jpg"
            try:
                file_path.parent.mkdir(exist_ok=True, parents=True)
                urllib.request.urlretrieve(image_url, file_path)
            except Exception as error:
                self.logger.warning(f"Failed to get image from link: {error}. Skip visualization step.")
                return
            return

        image = cv2.imread(str(file_path))
        tensor = self.preprocess(image)
        output = self.infer(tensor, asynchronous=True)
        boxes, scores, classes = self.postprocess(output, image.shape[:2])
        if len(boxes):
            for idx in range(tensor.shape[0]):
                imshow_det_bboxes(
                    image, np.concatenate([boxes[idx], scores[idx]], axis=1), classes[idx], self.class_names,
                    bbox_color=(0, 233, 255), text_color=(0, 233, 255), thickness=2, show=True, win_name=self.backend,
                )
