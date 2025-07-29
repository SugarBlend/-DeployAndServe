import json
from enum import Enum

import yaml
from pydantic import BaseModel, Field


class Formats(str, Enum):
    VIDEO = "video"
    IMAGE = "image"


class ProtocolType(str, Enum):
    GRPC = "grpc"
    HTTP = "http"


class Url(BaseModel):
    host: str = Field(description="Unique device name.")
    port: int = Field(description="Port number.")

    def __init__(self, host: str, port: int) -> None:
        super().__init__(host=host, port=port)

    def get_url(self) -> str:
        return f"{self.host}:{self.port}"


class OverrideFunctionality(BaseModel):
    module: str = Field(description="Way to module which consider override class.")
    cls: str = Field(description="Class which consist of overrides under basic functionality.")


class ServiceConfig(BaseModel):
    fastapi: Url = Field(description="Url for creation fastapi user server.")
    triton: Url = Field(description="Url for connection to triton container server.")
    protocol: ProtocolType = Field(description="Triton network protocol type.")
    server: OverrideFunctionality = Field(description="Module and class which implements server logic.")

    @classmethod
    def from_file(cls, path: str) -> "ServiceConfig":
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
        elif path.endswith(".yml") or path.endswith(".yaml"):
            with open(path, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
        else:
            raise NotImplementedError("At now support configuration files with such extensions: '.json', '.yml'.")

        return ServiceConfig.model_validate(data)
