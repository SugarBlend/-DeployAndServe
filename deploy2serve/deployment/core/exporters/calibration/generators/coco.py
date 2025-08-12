from collections import defaultdict
from pathlib import Path
from pycocotools.coco import COCO
from tqdm import tqdm
from typing import Any, Union, Dict, List

from deploy2serve.deployment.core.exporters.calibration.generators.interface import LabelsGenerator


class CocoGenerator(LabelsGenerator):
    def __init__(self, dataset_folder: Union[str, Path]) -> None:
        super().__init__(dataset_folder)

    def generate_labels(self) -> Dict[str, Any]:
        detections = defaultdict(list)
        anns_by_img = defaultdict(list)
        files: List[Path] = []

        annotations_json = self.dataset_folder.joinpath("annotations/person_keypoints_val2017.json")
        coco = COCO(annotations_json)
        category_ids = coco.getCatIds(catNms=["person"])
        image_ids = coco.getImgIds(catIds=category_ids)

        images_info = {img["id"]: img for img in coco.loadImgs(image_ids)}
        all_ann_ids = coco.getAnnIds(imgIds=image_ids, catIds=category_ids)

        for ann in coco.loadAnns(all_ann_ids):
            anns_by_img[ann["image_id"]].append(ann)

        self.logger.debug("Fetching COCO labels (category: person)...")
        for img_id in tqdm(image_ids, desc="Fetch labels", **self.progress_options):
            img_info = images_info[img_id]
            anns = anns_by_img.get(img_id)
            if not anns:
                continue

            stem = Path(img_info["file_name"]).stem
            for ann in anns:
                x1, y1, w, h = ann["bbox"]
                detections[stem].append([x1, y1, x1 + w, y1 + h])

            files.append(self.dataset_folder.joinpath(f"images/{img_info['file_name']}"))
        self.logger.debug(f"Labels successfully fetched, length of dataset: {len(files)}.")
        return {"files": files, "detections": detections}
