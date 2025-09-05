from abc import ABC, abstractmethod
from typing import Any


class ChunkCache(ABC):
    @abstractmethod
    def get(self, node: str, key: int) -> Any:
        pass

    @abstractmethod
    def put(self, node: str, key: int, value: Any) -> None:
        pass
