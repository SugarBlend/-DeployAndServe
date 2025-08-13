from argparse import Namespace
import cv2
from mmpose.visualization.fast_visualizer import FastVisualizer
import numpy as np
import torch
from typing import Tuple, Optional
import urllib.request

from deploy2serve.deployment.core.executors.factory import ExtendExecutor
from deploy2serve.deployment.models.export import ExportConfig
from deploy2serve.deployment.overrides.sapiens.model.postprocess import udp_decode
from deploy2serve.deployment.overrides.sapiens.model.preprocess import PosePreprocessor
from deploy2serve.deployment.overrides.sapiens.utils.adapters import visualizer_adapter
from deploy2serve.deployment.overrides.sapiens.utils.palettes import (
    COCO_KPTS_COLORS,
    COCO_SKELETON_INFO,
    COCO_WHOLEBODY_KPTS_COLORS,
    COCO_WHOLEBODY_SKELETON_INFO,
    GOLIATH_KPTS_COLORS,
    GOLIATH_SKELETON_INFO,
)
from deploy2serve.utils.logger import get_project_root


class SapiensExecutor(ExtendExecutor):
    def __init__(self, config: ExportConfig) -> None:
        super().__init__(config)
        self.batched_data: Optional[torch.Tensor] = None
        self.scales: Optional[torch.Tensor] = None
        self.centers: Optional[torch.Tensor] = None

        self.dtype = torch.float16 if self.config.enable_mixed_precision else torch.float32
        shape: Tuple[int, int] = self.config.input_shape[::-1] # type: ignore[attr-defined]
        self.preprocessor = PosePreprocessor(
            shape, torch.tensor([123.675, 116.28, 103.53]), torch.tensor([58.395, 57.12, 57.375]),
        )

    def preprocess(self, tensor: torch.Tensor, bboxes: torch.Tensor) -> None:
        self.batched_data, self.centers, self.scales = self.preprocessor(tensor.to(self.config.device),
                                                                         bboxes.to(self.config.device))

    def postprocess(self, heatmaps: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        joints, keypoint_scores = udp_decode(
            heatmaps[0].float().cpu().numpy(), # type: ignore[attr-defined]
            self.config.input_shape,
            np.array(self.config.input_shape) / 4,
            )
        joints = ((joints / self.preprocessor.input_shape) * self.scales[0] + self.centers[0] - 0.5 * self.scales[0])
        return joints, keypoint_scores

    @torch.inference_mode()
    def plotter(self) -> None:
        image_path = get_project_root().joinpath("resources/human-pose.jpg")
        if not image_path.exists():
            self.logger.warning(f"Image not found at this path: {image_path}. Try to download default image.")
            image_url = "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg"
            try:
                image_path.parent.mkdir(exist_ok=True, parents=True)
                urllib.request.urlretrieve(image_url, image_path) # noqa: S310
            except (Exception, ):
                self.logger.warning(f"Demo file is not exist: {image_path}, skip visualization step")
                return

        image = cv2.imread(str(image_path))
        h, w, c = image.shape
        tensor = torch.from_numpy(image).cuda().to(torch.float32).permute(2, 0, 1)
        self.preprocess(tensor, torch.tensor([[0, 0, w, h]]))
        output = self.infer(self.batched_data.to(self.dtype), asynchronous=False)[0]
        joints, keypoint_scores = self.postprocess(output)

        batch_size, num_joints = keypoint_scores.shape
        if num_joints == 308:  # noqa: PLR2004
            skeleton_info, pts_colors = GOLIATH_SKELETON_INFO, GOLIATH_KPTS_COLORS
        elif num_joints == 133:  # noqa: PLR2004
            skeleton_info, pts_colors = COCO_WHOLEBODY_SKELETON_INFO, COCO_WHOLEBODY_KPTS_COLORS
        else:
            skeleton_info, pts_colors = COCO_SKELETON_INFO, COCO_KPTS_COLORS

        meta_info = visualizer_adapter(skeleton_info, pts_colors)
        visualizer = FastVisualizer(meta_info, radius=3, line_width=1, kpt_thr=0.3)

        visualizer.draw_pose(image, Namespace(keypoints=joints, keypoint_scores=keypoint_scores))
        cv2.imshow("Visualization", image)
        cv2.waitKey(0)
