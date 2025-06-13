from argparse import ArgumentParser, Namespace
from importlib import import_module
import time
from typing import Type, Any

from triton.service import Service
from triton.configs import ServiceConfig


def parse_options() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--service_config", default="custom/yolo/config.json", help="Path to service configuration")
    return parser.parse_args()


def get_callable_from_string(path: str) -> Type[Any]:
    module_path, name = path.split(':')
    module = import_module(module_path)
    return getattr(module, name)


if __name__ == "__main__":
    args = parse_options()
    config = ServiceConfig.from_file(args.service_config)
    service = Service(
        inference_server_cls=get_callable_from_string(config.server),
        fastapi=config.fastapi,
        triton=config.triton,
        protocol=config.protocol,
        show=config.show,
        visualize_func=get_callable_from_string(config.visualize)
    )
    while service.runner.thread.is_alive():
        time.sleep(0.1)
