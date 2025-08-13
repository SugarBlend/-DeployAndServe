import math
import kornia
import numpy as np
import torch
from typing import Tuple


class PosePreprocessor:
    def __init__(
            self,
            input_shape: Tuple[int, int],
            mean: torch.Tensor,
            std: torch.Tensor,
            rot: float = 0.,
            device: str = "cuda:0",
    ) -> None:
        self.input_shape: Tuple[int, int] = input_shape
        self.mean: torch.Tensor = mean.view(-1, 1, 1).to(torch.float32).to(device)
        self.std: torch.Tensor = std.view(-1, 1, 1).to(torch.float32).to(device)
        self.rot: float = rot
        self.device: torch.device = torch.device(device)

    def __call__(
            self,
            img: torch.Tensor,
            bboxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        tensors, centers, scales = self.top_down_affine_transform(img, bboxes)
        tensors = kornia.geometry.transform.resize(tensors, self.input_shape[::-1])
        tensors = tensors[:, [2, 1, 0], ...]
        tensors = (tensors - self.mean) / self.std
        return tensors, centers.cpu().numpy(), scales.cpu().numpy()

    def top_down_affine_transform(
            self,
            img: torch.Tensor,
            bbox: torch.Tensor,
            padding: float = 1.25,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dim = bbox.ndim
        if dim == 1:
            bbox = bbox[None, :]

        x1, y1, x2, y2 = torch.hsplit(bbox, [1, 2, 3])
        center = torch.hstack([x1 + x2, y1 + y2]) * 0.5
        scale = torch.hstack([x2 - x1, y2 - y1]) * padding

        if dim == 1:
            center = center[0]
            scale = scale[0]

        w, h = self.input_shape
        aspect_ratio = w / h

        box_w, box_h = torch.hsplit(scale, [1])
        scale = torch.where(box_w > box_h * aspect_ratio,
                            torch.hstack([box_w, box_w / aspect_ratio]),
                            torch.hstack([box_h * aspect_ratio, box_h]))

        warp_mat = self.get_udp_warp_matrix(center, scale, output_size=(w, h))
        warp_img = kornia.geometry.transform.warp_affine(img[None].repeat_interleave(warp_mat.shape[0], dim=0),
                                                         warp_mat, (h, w))
        return warp_img, center, scale

    def get_udp_warp_matrix(
            self,
            center: torch.Tensor,
            scale: torch.Tensor,
            output_size: Tuple[int, int],
    ) -> torch.Tensor:
        rot_rad = math.radians(self.rot)
        cos_r = math.cos(rot_rad)
        sin_r = math.sin(rot_rad)

        scale_x = (output_size[0] - 1) / scale[:, 0]
        scale_y = (output_size[1] - 1) / scale[:, 1]

        input_size = center * 2
        tx = scale_x * (-0.5 * input_size[:, 0] * cos_r + 0.5 * input_size[:, 1] * sin_r +
                        0.5 * scale[:, 0])
        ty = scale_y * (-0.5 * input_size[:, 0] * sin_r - 0.5 * input_size[:, 1] * cos_r +
                        0.5 * scale[:, 1])

        return torch.stack([
            torch.stack([cos_r * scale_x, -sin_r * scale_x, tx], dim=1),
            torch.stack([sin_r * scale_y, cos_r * scale_y, ty], dim=1),
        ], dim=1)
