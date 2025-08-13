import numpy as np
from typing import Any, Dict, List, Tuple


def visualizer_adapter(skeleton_info: Dict[str, Any], pts_colors: List[List[int]]) -> Dict[str, Any]:
    pts_colors = np.asarray(pts_colors, dtype=np.uint8)
    skeleton_links: List[Tuple[int, int]] = []
    skeleton_link_colors: List[np.ndarray] = []

    for info in skeleton_info.values():
        pt1, pt2 = info["link"]
        color = np.asarray(info["color"], dtype=np.uint8)
        skeleton_links.append((pt1, pt2))
        skeleton_link_colors.append(color)

    return {
        "keypoint_id2name": None,
        "keypoint_name2id": None,
        "keypoint_colors": pts_colors,
        "skeleton_links": skeleton_links,
        "skeleton_link_colors": skeleton_link_colors,
    }
