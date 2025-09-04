from pathlib import Path
from typing import Generator, List, Optional
import tensorrt as trt
import torch
from tqdm import tqdm

from deploy2serve.deployment.core.exporters.calibration.batcher import BaseBatcher
from deploy2serve.deployment.models.export import TensorrtConfig
from deploy2serve.utils.logger import get_logger
from deploy2serve.deployment.utils.progress_utils import get_progress_options


class EngineCalibrator(trt.IInt8Calibrator):
    def __init__(
        self,
        config: TensorrtConfig,
        cache_path: str,
    ) -> None:
        super(EngineCalibrator, self).__init__()
        self.config: TensorrtConfig = config
        self.algorithm = self.config.specific.algorithm

        self.progress_bar: Optional[tqdm] = None
        self.image_batcher: Optional[BaseBatcher] = None
        self.batch_tensor: Optional[torch.Tensor] = None
        self.batch_generator: Optional[Generator[torch.Tensor]] = None

        self.logger = get_logger(self.__class__.__name__)
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def set_image_batcher(self, image_batcher: BaseBatcher) -> None:
        self.image_batcher = image_batcher
        self.batch_generator = self.image_batcher.get_batch()

    def get_batch_size(self) -> Optional[int]:
        if self.image_batcher:
            return self.image_batcher.batch_size
        return None

    def get_batch(self, names: List[str], p_str=None) -> Optional[List[int]]:
        if not self.image_batcher:
            return None

        if self.progress_bar is None and self.config.specific.log_level.value < 3:
            self.progress_bar = tqdm(total=self.image_batcher.total_frames, desc="INT8 Calibration",
                                     **get_progress_options())

        try:
            items = next(self.batch_generator)
            if self.progress_bar:
                self.progress_bar.update()
            return [int(item.contiguous().data_ptr()) for item in items]
        except StopIteration:
            if self.progress_bar:
                self.progress_bar.close()
            self.logger.info("Finished calibration step ...")
            return None

    def get_algorithm(self) -> trt.CalibrationAlgoType:
        return self.algorithm

    def read_calibration_cache(self) -> Optional[bytes]:
        if self.cache_path.exists() and self.config.enable_calibration_cache:
            self.logger.info(f"Using calibration cache file: {self.cache_path}")
            with self.cache_path.open("rb") as file:
                return file.read()
        return None

    def write_calibration_cache(self, cache: memoryview) -> None:
        self.logger.info(f"Writing calibration cache data to: {self.cache_path}")
        with self.cache_path.open("wb") as file:
            file.write(cache)
