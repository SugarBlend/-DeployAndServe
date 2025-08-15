import cv2
import numpy as np
from typing import Tuple


def gaussian_blur(heatmaps: np.ndarray, kernel: int = 11) -> np.ndarray:
    """Modulate heatmap distribution with Gaussian.

    Args:
        heatmaps (np.ndarray[num_joints, h, w]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (num_joints) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([num_joints, h, w]): Modulated heatmap distribution.

    """
    assert kernel % 2 == 1 # noqa: S101

    border = (kernel - 1) // 2
    num_joints, h, w = heatmaps.shape

    for k in range(num_joints):
        origin_max = np.max(heatmaps[k])
        dr = np.zeros((h + 2 * border, w + 2 * border), dtype=np.float32)
        dr[border:-border, border:-border] = heatmaps[k].copy()
        dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
        heatmaps[k] = dr[border:-border, border:-border].copy()
        heatmaps[k] *= origin_max / np.max(heatmaps[k])
    return heatmaps


def get_heatmap_maximum(heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from heatmaps.

    Args:
        heatmaps (np.ndarray): Heatmaps in shape (num_joints, h, w) or (batch_size, num_joints, h, w)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (num_joints, 2) or (batch_size, num_joints, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (num_joints,) or (batch_size, num_joints)

    """
    assert isinstance(heatmaps, np.ndarray), "heatmaps should be numpy.ndarray" # noqa: S101
    assert heatmaps.ndim in {3, 4}, f"Invalid shape {heatmaps.shape}" # noqa: S101

    if heatmaps.ndim == 3: # noqa: PLR2004
        num_joints, h, w = heatmaps.shape
        batch_size = None
        heatmaps_flatten = heatmaps.reshape(num_joints, -1)
    else:
        batch_size, num_joints, h, w = heatmaps.shape
        heatmaps_flatten = heatmaps.reshape(batch_size * num_joints, -1)

    y_locs, x_locs = np.unravel_index(np.argmax(heatmaps_flatten, axis=1), shape=(h, w))
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vals = np.amax(heatmaps_flatten, axis=1)
    locs[vals <= 0.] = -1

    if batch_size:
        locs = locs.reshape(batch_size, num_joints, 2)
        vals = vals.reshape(batch_size, num_joints)

    return locs, vals


def refine_keypoints_dark_udp(keypoints: np.ndarray, heatmaps: np.ndarray, blur_kernel_size: int) -> np.ndarray:
    """Refine keypoint predictions using distribution aware coordinate decoding
    for UDP. See `UDP`_ for details. The operation is in-place.

    Args:
        keypoints (np.ndarray): The keypoint coordinates in shape (N, K, D)
        heatmaps (np.ndarray): The heatmaps in shape (K, H, W)
        blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
            modulation

    Returns:
        np.ndarray: Refine keypoint coordinates in shape (N, K, D)

    .. _`UDP`: https://arxiv.org/abs/1911.07524

    """
    num_instances, num_joints = keypoints.shape[:2]
    h, w = heatmaps.shape[1:]

    # modulate heatmaps
    heatmaps = gaussian_blur(heatmaps, blur_kernel_size)
    np.clip(heatmaps, 1e-3, 50., heatmaps)
    np.log(heatmaps, heatmaps)

    heatmaps_pad = np.pad(heatmaps, ((0, 0), (1, 1), (1, 1)), mode="edge").flatten()

    for n in range(num_instances):
        index = keypoints[n, :, 0] + 1 + (keypoints[n, :, 1] + 1) * (w + 2)
        index += (w + 2) * (h + 2) * np.arange(0, num_joints)
        index = index.astype(int).reshape(-1, 1)
        i_ = heatmaps_pad[index]
        ix1 = heatmaps_pad[index + 1]
        iy1 = heatmaps_pad[index + w + 2]
        ix1y1 = heatmaps_pad[index + w + 3]
        ix1_y1_ = heatmaps_pad[index - w - 3]
        ix1_ = heatmaps_pad[index - 1]
        iy1_ = heatmaps_pad[index - 2 - w]

        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = np.concatenate([dx, dy], axis=1)
        derivative = derivative.reshape(num_joints, 2, 1)

        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
        hessian = hessian.reshape(num_joints, 2, 2)
        hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        keypoints[n] -= np.einsum("imn,ink->imk", hessian,
                                  derivative).squeeze()

    return keypoints


def udp_decode(
        heatmaps: np.ndarray,
        input_size: Tuple[int, int],
        heatmap_size: Tuple[int, int],
        blur_kernel_size: int = 11,
) -> Tuple[np.ndarray, np.ndarray]:
    keypoints, scores = get_heatmap_maximum(heatmaps)
    keypoints = keypoints[None]
    scores = scores[None]
    keypoints = refine_keypoints_dark_udp(keypoints, heatmaps, blur_kernel_size=blur_kernel_size)

    w, h = heatmap_size
    keypoints = (keypoints / [w - 1, h - 1]) * input_size
    return keypoints, scores
