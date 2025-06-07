from deployment.core.exporters.base import BaseExporter, ExportConfig


class OpenVINOExporter(BaseExporter):
    def __init__(self, config: ExportConfig):
        super(OpenVINOExporter, self).__init__(config)

    def export(self) -> None:
        pass

    def benchmark(self) -> None:
        pass
