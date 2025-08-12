from abc import ABC, abstractmethod
from typing import Any


class ChunkCache(ABC):
    @abstractmethod
    def get(self, key: int) -> Any:
        pass

    @abstractmethod
    def put(self, key: int, value: Any) -> None:
        pass
