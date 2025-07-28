import os

import cv2
import numpy as np
from mmcv.visualization.image import imshow_det_bboxes
import torch
from typing import Any, List, Tuple, Optional
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import non_max_suppression, scale_boxes

from triton.base import TritonRemote, ProtocolType
from utils.containers import is_image_file, is_video_file

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

    def postprocess(self, output: np.ndarray) -> np.ndarray:
        output = torch.from_numpy(output[0]).cuda()
        detections = non_max_suppression(output)[0]
        boxes = scale_boxes(self.model_shape, detections[:, :4], self.input_shape).cpu().numpy()
        scores = detections[:, 4:5].reshape(-1, 1).cpu().numpy()
        classes = detections[:, 5:].reshape(-1, 1).to(torch.int).cpu().numpy()
        return np.concatenate([boxes, scores, classes], axis=1)


def plot(frame: np.ndarray, outputs: np.ndarray, class_names: Optional[List[str]] = None, wait_time: int = 0):
    if len(outputs):
        cv2.namedWindow("triton", cv2.WINDOW_GUI_EXPANDED)
        imshow_det_bboxes(frame, outputs[:, :5], outputs[:, 5],
                          class_names, bbox_color=(0, 233, 255), text_color=(0, 233, 255), thickness=2,
                          show=True, win_name="triton", wait_time=wait_time)


def check_labels(source_path: str, labels: str) -> None:
    import pickle
    with open(labels, "rb") as file:
        detections = pickle.load(file)

    if is_video_file(source_path):
        from tqdm import trange
        from imutils.video import FileVideoStream
        cap = FileVideoStream(source_path)
        cap.start()
        frames_number = int(cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in trange(frames_number, desc="Read frames"):
            frame = cap.read()
            if frame is None:
                break
            plot(frame, np.array(detections[i]).reshape(-1, 6), wait_time=10)
    elif is_image_file(source_path):
        frame = cv2.imread(source_path)
        plot(frame, np.array(detections[0]).reshape(-1, 6), wait_time=0)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    check_labels("../../resources/cup.mp4", "../../resources/cup.pickle")
    check_labels("../../resources/demo.jpg", "../../resources/demo.pickle")
