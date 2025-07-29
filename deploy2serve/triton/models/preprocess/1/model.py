import io
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import triton_python_backend_utils as pb_utils  # type: ignore[attr-defined]
from PIL import Image


class TritonPythonModel:
    def __init__(self) -> None:
        self.target_height: Optional[int] = None
        self.target_width: Optional[int] = None
        self.device: Optional[str] = None
        self.outputs: Optional[Dict[str, Any]] = None
        self.inputs: Optional[Dict[str, Any]] = None
        self.config: Optional[Dict[str, Any]] = None

    def initialize(self, args: Any) -> None:
        self.config = json.loads(args["model_config"])
        self.inputs = {}
        self.outputs = {}

        for node in self.config["input"]:
            self.inputs[node["name"]] = {
                "dtype": pb_utils.triton_string_to_numpy(node["data_type"]),
                "dims": node["dims"],
            }
        for node in self.config["output"]:
            self.outputs[node["name"]] = {
                "dtype": pb_utils.triton_string_to_numpy(node["data_type"]),
                "dims": node["dims"],
            }

        self.device = "cuda" if args["model_instance_kind"] == "GPU" else "cpu"
        self.target_width = 640
        self.target_height = 384

    def preprocess_image(self, image: Image) -> Tuple[np.ndarray, np.ndarray]:
        image = image.convert("RGB")
        width, height = image.size

        scale = min(self.target_width / width, self.target_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)

        padded_image = np.full((self.target_height, self.target_width, 3), 114, dtype=np.float32)
        dw = (self.target_width - new_width) // 2
        dh = (self.target_height - new_height) // 2
        padded_image[dh : dh + new_height, dw : dw + new_width] = np.array(image)

        tensor = padded_image.transpose(2, 0, 1) / 255.0
        return tensor[None].astype(np.float16), np.array([height, width], dtype=np.float16).reshape(-1, 2)

    def execute(self, requests: List["pb_utils.InferenceRequest"]) -> List["pb_utils.InferenceResponse"]:
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_0")
            image_data = input_tensor.as_numpy().tobytes()
            image = Image.open(io.BytesIO(image_data))

            processed_tensor, original_shape = self.preprocess_image(image)

            if len(self.outputs) != 2:
                raise ValueError("Expected exactly 2 output tensors: preprocessed_image and input_shape")
            tensor_node, shape_node = self.outputs.keys()

            tensor_out = torch.from_numpy(processed_tensor).to(self.device)
            output_tensor = pb_utils.Tensor(tensor_node, tensor_out.cpu().numpy())

            shape_out = torch.from_numpy(original_shape).to(self.device)
            output_shape = pb_utils.Tensor(shape_node, shape_out.cpu().numpy())

            responses.append(pb_utils.InferenceResponse([output_tensor, output_shape]))
        return responses
