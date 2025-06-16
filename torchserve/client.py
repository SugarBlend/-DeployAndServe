import cv2
import requests
from pathlib import Path
from typing import Any, Dict
import numpy as np
from abc import ABC, abstractmethod
from argparse import Namespace, ArgumentParser
from mmcv.visualization.image import imshow_det_bboxes


class SimpleAPI(object):
    def __init__(self) -> None:
        # self.show: bool = args.show
        pass

    @staticmethod
    def _request(data, model_name) -> Dict[str, Any]:
        response = requests.post(
            f"http://localhost:8080/predictions/{model_name}",
            data=data
        )
        return response.json()

    def _image_request(self, image_path: str, model_name: str = "yolo") -> Dict[str, Any]:
        try:
            with open(image_path, 'rb') as file:
                data = file.read()
            results = self._request(data, model_name)
            return results
        except Exception as error:
            return {"error": error}

    # async def _video_request(self, source_path: str, model_name: str = "yolo") -> Dict[str, Any]:
    #     try:
    #         cap = cv2.VideoCapture(source_path)
    #         while True:
    #             ret, image = cap.read()
    #             _, buffer = cv2.imencode('.jpg', image)
    #             data = buffer.tobytes()
    #
    #             results = await self._request(data, model_name)
    #             if self.show:
    #                 self.visualize()
    #         return results
    #     except Exception as error:
    #         return {"error": error}

if __name__ == "__main__":
    api = SimpleAPI()
    image_path = f"{Path(__file__).parents[1]}/deployment/resources/demo.jpg"
    results = api._image_request(image_path, "yolo")
    boxes = []
    scores = []
    classes = []
    for item in results["predictions"]:
        boxes.append(item["bbox"])
        scores.append(item["score"])
        classes.append(item["class"])
    boxes, scores, classes = np.array(boxes).reshape(-1, 4), np.array(scores).reshape(-1, 1), np.array(classes)
    imshow_det_bboxes(cv2.imread(image_path), np.concatenate([boxes, scores], axis=1), classes,
                      bbox_color=(0, 233, 255), text_color=(0, 233, 255), thickness=2,
                      show=True, win_name="torchserve", wait_time=0)