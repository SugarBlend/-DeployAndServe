from argparse import ArgumentParser, Namespace
from importlib import import_module
from typing import Callable
from pathlib import Path
from models.export import ExportConfig

from deploy2serve.utils.logger import get_project_root


def parse() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--deploy_config", default="overrides/yolo/configs/efficient_nms.yml",
                        type=str, help="Way for deploy configuration.")
    return parser.parse_args()


def get_object(module_name: str, cls_name: str) -> Callable:
    return getattr(import_module(module_name), cls_name)


def converter(args: Namespace) -> None:
    config = ExportConfig.from_file(args.deploy_config)

    exporter = get_object(config.exporter.module, config.exporter.cls)(config)

    if not Path(config.torch_weights).is_absolute():
        config.torch_weights = str(get_project_root().joinpath(config.torch_weights))

    exporter.load_checkpoints(config.torch_weights)
    executor = get_object(config.executor.module, config.executor.cls)(config)

    for backend in config.formats:
        exporter.convert(backend)
        if config.enable_visualization:
            executor.visualization(backend)


if __name__ == "__main__":
    converter(parse())
