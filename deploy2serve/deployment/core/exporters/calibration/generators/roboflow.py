import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Any, Union, Dict
from mmengine.config import Config
from deploy2serve.deployment.core.exporters.calibration.generators.interface import LabelsGenerator
from deploy2serve.utils.containers import is_image_file


class RoboflowGenerator(LabelsGenerator):
    def __init__(self, dataset_folder: Union[str, Path]) -> None:
        super().__init__(dataset_folder)

    def generate_labels(self) -> Dict[str, Any]:
        roboflow_config = self.dataset_folder.joinpath("data.yaml")
        if not roboflow_config.exists():
            raise Exception("The data index file should be in the preloaded dataset from the Roboflow platform, but "
                            "it was not found.")

        dataset_config = Config.fromfile(roboflow_config)
        sections = ["train", "val", "test"]
        for field in sections:
            if hasattr(dataset_config, field):
                images_path = Path(getattr(dataset_config, field))
                if images_path.exists():
                    images = images_path.glob("*.*")
                    labels = list(images_path.parent.joinpath("labels").glob("*.txt"))
                    break
        else:
            raise Exception("Didn't found basic folders with separated data of dataset.")

        detections: Dict[str, np.ndarray] = {}

        images = [item for item in images if is_image_file(item)]
        if not len(images):
            raise Exception(
                f"Folder '{self.dataset_folder}' doesn't have files with mimetype 'image'. Check you dataset.",
            )
        if not len(labels):
            raise Exception(
                f"Folder: '{self.dataset_folder}/annotations' doesn't have annotation files in '.txt' "
                f"format. Check your labels.",
            )

        image_files = {img_path.stem: img_path for img_path in images}

        for description_path in tqdm(labels, desc="Fetch labels", **self.progress_options):
            stem = description_path.stem
            img_path = image_files.get(stem)
            if not img_path:
                continue

            try:
                with Image.open(img_path) as img:
                    w, h = img.size
            except Exception as error:
                self.logger.warning(error)
                continue

            with description_path.open() as file:
                for line in file:
                    if not line.strip():
                        continue
                    cls, center_x, center_y, width, height = map(float, line.split())
                    x1, y1 = (center_x - width / 2) * w, (center_y - height / 2) * h
                    x2, y2 = (center_x + width / 2) * w, (center_y + height / 2) * h
                    detections.setdefault(stem, []).append(np.array([x1, y1, x2, y2], dtype=np.int32))

        files = sorted(image_files.values())
        self.logger.debug(f"Labels successfully fetched, length of dataset: {len(files)}.")
        return {"files": files, "detections": detections}
