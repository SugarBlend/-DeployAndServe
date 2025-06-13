from abc import ABC, abstractmethod
import numpy as np
from tritonclient.utils import triton_to_np_dtype
from typing import Optional, List, Dict, Any

from configs import ProtocolType


class TritonRemote(ABC):
    def __init__(self, url: str, model_name: str, protocol: ProtocolType) -> None:
        super(TritonRemote, self).__init__()
        self.url: str = url
        self.model_name: str = model_name
        self.protocol: ProtocolType = protocol

        self.triton_client: Optional["client.InferenceServerClient"] = None
        self.metadata: Optional["service_pb2.ModelMetadataResponse"] = None
        self.inputs: Dict[str, "client.InferInput"] = {}
        self.outputs: Dict[str, "client.InferRequestedOutput"] = {}

    async def initialize(self) -> None:
        if self.triton_client is None:
            options = {}
            if self.protocol == ProtocolType.GRPC:
                import tritonclient.grpc.aio as client
                options.update(dict(as_json=True))
            else:
                import tritonclient.http.aio as client

            self.triton_client = client.InferenceServerClient(self.url, verbose=False)
            self.metadata = await self.triton_client.get_model_metadata(self.model_name, **options)

            for node in self.metadata["inputs"]:
                shape = list(map(int, node["shape"]))
                self.inputs.update({node["name"]: client.InferInput(node["name"], [1, *shape[1:]], node["datatype"])})
            self.outputs = {node["name"]: client.InferRequestedOutput(node["name"]) for node in self.metadata["outputs"]}

    async def infer(self, feed_input: Dict[str, np.ndarray]) -> List[np.ndarray]:
        for node in feed_input:
            triton_type = triton_to_np_dtype(self.inputs[node].datatype())
            self.inputs[node].set_data_from_numpy(feed_input[node].astype(triton_type))

        results = await self.triton_client.infer(model_name=self.model_name, inputs=self.inputs.values(),
                                                 outputs=self.outputs.values())
        outputs = [results.as_numpy(node["name"]) for node in self.metadata["outputs"]]
        return outputs

    @abstractmethod
    def preprocess(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def postprocess(self, *args, **kwargs) -> Any:
        pass
