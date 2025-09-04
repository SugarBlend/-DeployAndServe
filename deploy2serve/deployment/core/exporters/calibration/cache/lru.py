from collections import OrderedDict
from typing import Any, Dict

from deploy2serve.deployment.core.exporters.calibration.cache.interface import ChunkCache


class LRUChunkCache(ChunkCache):
    def __init__(self, max_chunks: int = 4) -> None:
        self.max_chunks: int = max_chunks
        self.cache: Dict[str, OrderedDict[int, Any]] = {}

    def get(self, node: str, key: int) -> Any:
        node_cache = self.cache.get(node)
        if node_cache is None or key not in node_cache:
            return None
        node_cache.move_to_end(key)
        return node_cache[key]

    def put(self, node: str, key: int, value: Any) -> None:
        node_cache = self.cache.setdefault(node, OrderedDict())
        node_cache[key] = value
        node_cache.move_to_end(key)
        if len(node_cache) > self.max_chunks:
            node_cache.popitem(last=False)
