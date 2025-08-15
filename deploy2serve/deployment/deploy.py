from argparse import ArgumentParser, Namespace
from importlib import import_module
from pathlib import Path
from typing import Callable

from deploy2serve.deployment.models.export import ExportConfig


def parse() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--deploy_config", default="projects/yolo/configs/dynamic.yml", type=str, help="Way for deploy configuration.")
    return parser.parse_args()


def get_object(module_name: str, cls_name: str) -> Callable:
    return getattr(import_module(module_name), cls_name)


def converter(args: Namespace) -> None:
    config = ExportConfig.from_file(args.deploy_config)

    exporter = get_object(config.exporter.module_path, config.exporter.class_name)(config)

    if not Path(config.torch_weights).is_absolute():
        config.torch_weights = str(Path.cwd().joinpath(config.torch_weights))

    exporter.load_checkpoints(config.torch_weights, config.model_configuration)
    executor = get_object(config.executor.module_path, config.executor.class_name)(config)

    for backend in config.formats:
        exporter.convert(backend)
        if config.enable_visualization:
            executor.visualization(backend)


if __name__ == "__main__":
    converter(parse())
