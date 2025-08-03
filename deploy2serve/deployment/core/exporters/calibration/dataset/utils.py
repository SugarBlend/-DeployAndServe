from collections import OrderedDict
from typing import Any


class LRUChunkCache(object):
    def __init__(self, max_chunks: int = 4) -> None:
        self.max_chunks: int = max_chunks
        self.cache: OrderedDict[int, Any] = OrderedDict()

    def get(self, key: int) -> Any:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: int, value: Any) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.max_chunks:
            self.cache.popitem(last=False)
