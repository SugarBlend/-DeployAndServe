import cv2
import numpy as np
from mmcv.visualization.image import imshow_det_bboxes
import torch
from typing import Any, List, Tuple, Optional, Dict
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import non_max_suppression, scale_boxes

from triton.base import TritonRemote, ProtocolType


class EnsembleYoloTriton(TritonRemote):
    def __init__(self, url: str, model_name: str, protocol: ProtocolType) -> None:
        super(EnsembleYoloTriton, self).__init__(url, model_name, protocol)

    def preprocess(self, *args, **kwargs) -> Any:
        pass

    def postprocess(self, *args, **kwargs) -> Any:
        pass


class YoloTriton(TritonRemote):
    def __init__(self, url: str, model_name: str, protocol: ProtocolType) -> None:
        super(YoloTriton, self).__init__(url, model_name, protocol)
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

    def postprocess(self, output: np.ndarray) -> Dict[str, List[Any]]:
        output = torch.from_numpy(output[0]).cuda()
        detections = non_max_suppression(output)[0]
        boxes = scale_boxes(self.model_shape, detections[:, :4], self.input_shape).cpu().numpy()
        scores = detections[:, 4:5].reshape(-1, 1).cpu().numpy()
        classes = detections[:, 5:].reshape(-1).to(torch.int).cpu().numpy()

        return {"boxes": boxes.tolist(), "scores": scores.tolist(), "classes": classes.tolist()}


def visualize(frame: np.ndarray, outputs: Dict[str, List[Any]], class_names: Optional[List[str]] = None, wait_time: int = 0):
    boxes, scores, classes = outputs.values()
    if len(boxes):
        cv2.namedWindow("triton", cv2.WINDOW_GUI_EXPANDED)
        imshow_det_bboxes(frame, np.concatenate([boxes, scores], axis=1), np.array(classes),
                          class_names, bbox_color=(0, 233, 255), text_color=(0, 233, 255), thickness=2,
                          show=True, win_name="triton", wait_time=wait_time)
