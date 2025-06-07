from copy import deepcopy
import os
import onnx
import onnxslim
from pathlib import Path
import torch

from deployment.core.exporters import BaseExporter, ExportConfig
from utils.logger import get_logger


class ONNXExporter(BaseExporter):
    def __init__(self, config: ExportConfig) -> None:
        super(ONNXExporter, self).__init__(config)
        self.onnx_path = Path(self.config.onnx_opts.output_file)
        self.onnx_path.parent.mkdir(exist_ok=True, parents=True)
        self.logger = get_logger("onnx")

    def export(self) -> None:
        if os.path.exists(self.onnx_path) and not self.config.onnx_opts.force_rebuild:
            return

        self.logger.info("Try to convert Pytorch model to onnx format")
        model = deepcopy(self.model)
        device = torch.device(self.config.device)
        for p in model.parameters():
            p.required_grad = False
        model.eval()
        model.to(device)
        model.float()

        dummy_input = torch.zeros((1, 3, *self.config.input_shape), dtype=torch.float32, device=device)
        torch.onnx.export(
            model,
            (dummy_input,),
            self.onnx_path,
            **self.config.onnx_opts.model_dump()
        )

        onnx_model = onnx.load(self.onnx_path)
        onnx.checker.check_model(onnx_model)
        if self.config.onnx_opts.simplify:
            self.logger.info("Try to simplify Onnx model")
            onnxslim.slim(onnx_model, model_check=True, skip_optimizations=False, skip_fusion_patterns=False,
                          output_model=self.onnx_path)
            self.logger.info("Simplification successfully done")
        self.logger.info(f"Onnx model successfully stored in: {self.onnx_path}")

    def benchmark(self) -> None:
        pass
