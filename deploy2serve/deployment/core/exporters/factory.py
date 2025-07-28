import types
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, Dict, Optional, Type
import torch.nn

from deploy2serve.deployment.core.exporters import BaseExporter, ONNXExporter, OpenVINOExporter, TensorRTExporter, TorchScriptExporter
from deploy2serve.deployment.models.export import Backend, ExportConfig


class ExporterFactory(object):
    _exporters: Dict[Backend, Type[BaseExporter]] = {
        Backend.TorchScript: TorchScriptExporter,
        Backend.ONNX: ONNXExporter,
        Backend.TensorRT: TensorRTExporter,
        Backend.OpenVINO: OpenVINOExporter
    }

    @classmethod
    def create(cls, export_type: Backend) -> Type[BaseExporter]:
        exporter_class = cls._exporters.get(export_type)
        if not exporter_class:
            raise ValueError(f"Unsupported exporter type: {export_type}")
        return exporter_class


class Exporter(ABC):
    def __init__(self, config: ExportConfig) -> None:
        self.config: ExportConfig = config
        self.export_factory = ExporterFactory()
        self.model: Optional[torch.nn.Module] = None
        self.onnx_patch = nullcontext

    @abstractmethod
    def load_checkpoints(self, **kwargs) -> Any:
        pass

    def convert(self, backend: Backend) -> None:
        exporter = self.export_factory.create(backend)(self.config)

        if backend == Backend.TensorRT:
            if hasattr(self, "register_tensorrt_plugins"):
                exporter.register_tensorrt_plugins = types.MethodType(self.register_tensorrt_plugins.__func__, exporter)
        elif backend == Backend.ONNX:
            if hasattr(self, "register_onnx_plugins"):
                exporter.register_onnx_plugins = types.MethodType(self.register_onnx_plugins.__func__, exporter)

        exporter.__dict__.update(self.__dict__)
        exporter.export()

        if self.config.enable_benchmark:
            exporter.benchmark()
