from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy
from typing import Optional, List, Dict, Any, Union
from tritonclient.utils import triton_to_np_dtype

from deploy2serve.triton.core.configs import ProtocolType


class TritonRemote(ABC):
    def __init__(self, url: str, model_name: str, protocol: ProtocolType) -> None:
        super(TritonRemote, self).__init__()
        self.url: str = url
        self.model_name: str = model_name
        self.protocol: ProtocolType = protocol

        self.client: Optional[Union["tritonclient.grpc.aio", "tritonclient.http.aio"]] = None
        self.triton_client: Optional["client.InferenceServerClient"] = None
        self.metadata: Optional["service_pb2.ModelMetadataResponse"] = None
        self.inputs: Dict[str, "client.InferInput"] = {}
        self.outputs: Dict[str, "client.InferRequestedOutput"] = {}
        self.counter: int = 0

    async def initialize(self) -> None:
        if self.triton_client is None:
            options = {}
            if self.protocol == ProtocolType.GRPC:
                import tritonclient.grpc.aio as client
                options.update(dict(as_json=True))
            else:
                import tritonclient.http.aio as client

            self.client = client
            self.triton_client = client.InferenceServerClient(self.url, verbose=False)
            self.metadata = await self.triton_client.get_model_metadata(self.model_name, **options)

            self.outputs = {node["name"]: client.InferRequestedOutput(node["name"]) for node in self.metadata["outputs"]}

    async def infer(self, feed_input: List[np.ndarray]) -> List[np.ndarray]:
        inputs = []
        for idx, node in enumerate(self.metadata["inputs"]):
            triton_type = triton_to_np_dtype(node["datatype"])
            data = feed_input[idx].astype(triton_type)
            inputs.append(self.client.InferInput(node["name"], list(feed_input[idx].shape),
                                                 node["datatype"]).set_data_from_numpy(data))

        results = await self.triton_client.infer(model_name=self.model_name, inputs=inputs,
                                                 outputs=self.outputs.values(), request_id=f'request:{self.counter}')
        outputs = [deepcopy(results.as_numpy(node["name"])) for node in self.metadata["outputs"]]
        return outputs

    @abstractmethod
    def preprocess(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def postprocess(self, *args, **kwargs) -> Any:
        pass
