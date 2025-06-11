from models.export import ExportConfig
from deployment.custom.yolo.executor import YoloExecutor
from deployment.custom.yolo.export import YoloExporter
from argparse import ArgumentParser, Namespace


def parse() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--deploy_config", default="custom/yolo/configs/export_yolo_batched_nms.yml",
                        type=str, help="Way for deploy configuration.")
    parser.add_argument("--weights_path", default="weights/pytorch/yolo12m.pt", type=str,
                        help="Way for model weights.")
    return parser.parse_args()

def converter(args: Namespace) -> None:
    config = ExportConfig.from_file(args.deploy_config)

    exporter = YoloExporter(config)
    exporter.load_checkpoints(args.weights_path)
    executor = YoloExecutor(config)

    for backend in config.formats:
        exporter.convert(backend)
        if config.enable_visualization:
            executor.visualization(backend)


if __name__ == "__main__":
    converter(parse())
