from abc import ABC, abstractmethod
import torch.nn
from typing import Dict, Type, Any, Optional
import types

from deployment.core.exporters import OpenVINOExporter, BaseExporter, TensorRTExporter, ONNXExporter
from deployment.models.export import Backend, ExportConfig


class ExporterFactory(object):
    _exporters: Dict[Backend, Type[BaseExporter]] = {
        Backend.ONNX: ONNXExporter,
        Backend.TensorRT: TensorRTExporter,
        Backend.OpenVINO: OpenVINOExporter,
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
        self.model : Optional[torch.nn.Module] = None

    @abstractmethod
    def load_checkpoints(self, **kwargs) -> Any:
        pass

    def convert(self, backend: Backend) -> None:
        exporter = self.export_factory.create(backend)(self.config)

        # for k, v in self.__dict__.items():
        #     if not callable(v) and not k.startswith('__'):
        #         setattr(exporter, k, v)

            # 3. Специальная обработка для TensorRT
        if backend == Backend.TensorRT:
            if hasattr(self, 'register_tensorrt_plugins'):
                # Правильно привязываем метод к новому экземпляру
                exporter.register_tensorrt_plugins = types.MethodType(
                    self.register_tensorrt_plugins.__func__,  # Берем оригинальную функцию
                    exporter  # Привязываем к новому экземпляру
                )

        exporter.__dict__.update(self.__dict__)
        exporter.export()
        # self.__dict__.update(exporter.__dict__)
        self.__dict__.update({
            k: v for k, v in exporter.__dict__.items()
            if not callable(v) and not k.startswith('__')
        })

        if self.config.enable_benchmark:
            exporter.benchmark()
