from typing import Any, Union, Dict
from pathlib import Path

from deploy2serve.deployment.core.exporters.calibration.generators.interface import LabelsGenerator


class RoboflowGenerator(LabelsGenerator):
    def __init__(self, dataset_folder: Union[str, Path]) -> None:
        super().__init__(dataset_folder)

    def generate_labels(self) -> Dict[str, Any]:
        pass
