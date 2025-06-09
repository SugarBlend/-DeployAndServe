from models.export import ExportConfig

from deployment.custom.yolo.executor import YoloExecutor
from deployment.custom.yolo.export import YoloExporter


def converter() -> None:
    weights_path = "weights/pytorch/yolov5mu.pt"
    deploy_config = "configs/export_yolo.yml"
    config = ExportConfig.from_file(deploy_config)

    exporter = YoloExporter(config)
    exporter.load_checkpoints(weights_path)
    executor = YoloExecutor(config)

    for backend in config.pipeline:
        exporter.convert(backend)
        if config.enable_visualization:
            executor.visualization(backend)


if __name__ == "__main__":
    converter()
