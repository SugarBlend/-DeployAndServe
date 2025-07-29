from typing import List, Optional

import cv2
import numpy as np
from mmcv.visualization.image import imshow_det_bboxes

from deploy2serve.utils.containers import is_image_file, is_video_file


def plot(frame: np.ndarray, outputs: np.ndarray, class_names: Optional[List[str]] = None, wait_time: int = 0):
    if len(outputs):
        cv2.namedWindow("triton", cv2.WINDOW_GUI_EXPANDED)
        imshow_det_bboxes(
            frame,
            outputs[:, :5],
            outputs[:, 5],
            class_names,
            bbox_color=(0, 233, 255),
            text_color=(0, 233, 255),
            thickness=2,
            show=True,
            win_name="triton",
            wait_time=wait_time,
        )


def check_labels(source_path: str, labels: str) -> None:
    import pickle

    with open(labels, "rb") as file:
        detections = pickle.load(file)

    if is_video_file(source_path):
        from imutils.video import FileVideoStream
        from tqdm import trange

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
