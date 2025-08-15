from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union, Any, Generator, Type

import numpy as np
import tensorrt as trt
import torch
from ultralytics import YOLO

from deploy2serve.deployment.core.exporters.calibration.batcher import BaseBatcher
from deploy2serve.deployment.core.exporters.factory import Exporter
from deploy2serve.deployment.core.exporters.backends.onnx_format import ONNXExporter
from deploy2serve.deployment.core.exporters.backends.tensorrt_format import TensorRTExporter, ExporterFactory, Backend
from deploy2serve.deployment.models.export import ExportConfig
from deploy2serve.deployment.models.common import Plugin
from deploy2serve.deployment.projects.yolo.model import Model, WrappedModel
from deploy2serve.deployment.projects.yolo.batcher import DetectionBatcher


@ExporterFactory.register(Backend.ONNX)
class OverrideONNX(ONNXExporter):
    def __init__(self, config: ExportConfig, model: torch.nn.Module):
        super().__init__(config, model)

    @contextmanager
    def patch_ops(self) -> Generator[None, Any, None]:
        if self.config.enable_mixed_precision:
            func = torch.arange

            def arange(*args, dtype: Optional[torch.dtype] = None, **kwargs) -> torch.Tensor:
                return func(*args, **kwargs).to(dtype)

            torch.arange = arange
            yield
            torch.arange = func
        else:
            yield

    def register_onnx_plugins(self) -> Any:
        pass


@ExporterFactory.register(Backend.TensorRT)
class OverrideTensorRT(TensorRTExporter):
    def __init__(self, config: ExportConfig, model: torch.nn.Module):
        super().__init__(config, model)

    def register_batcher(self) -> Optional[Type[BaseBatcher]]:
        return DetectionBatcher(self.config, "yolo", self.config.input_shape)

    def register_tensorrt_plugins(self, network: trt.INetworkDefinition) -> trt.INetworkDefinition:
        available_plugins = {
            "efficient_nms": add_efficient_nms_plugin,
            "batched_nms": add_batched_nms_plugin,
        }

        for search_plugin, impl in available_plugins.items():
            plugin = next((plugin for plugin in self.config.tensorrt.plugins if plugin.name == search_plugin), None)
            if plugin:
                plugin.options.update({"nc": self.model.nc})
                network = impl(network, plugin)
        return network


def add_efficient_nms_plugin(network: trt.INetworkDefinition, plugin: Plugin) -> trt.INetworkDefinition:
    previous_output = network.get_output(0)
    network.unmark_output(previous_output)

    # Prepare tensors with better dimension handling
    # Transpose and reshape operations
    shuffle_layer = network.add_shuffle(previous_output)
    shuffle_layer.second_transpose = (0, 2, 1)

    # Get dimensions with error checking
    try:
        bs, num_boxes, temp = shuffle_layer.get_output(0).shape
    except Exception as error:
        raise ValueError(f"Invalid output shape: {error}")

    # Slice boxes and scores with explicit dimensions
    strides = trt.Dims([1, 1, 1])
    starts = trt.Dims([0, 0, 0])

    # Boxes extraction [x1,y1,x2,y2]
    boxes_shape = trt.Dims([bs, num_boxes, 4])
    boxes = network.add_slice(shuffle_layer.get_output(0), starts, boxes_shape, strides)

    # Scores extraction (after 4 box coordinates)
    starts[2] = 4
    scores_shape = trt.Dims([bs, num_boxes, temp - 4])
    scores = network.add_slice(shuffle_layer.get_output(0), starts, scores_shape, strides)

    # Create EfficientNMS plugin with validation
    registry = trt.get_plugin_registry()
    creator = registry.get_plugin_creator("EfficientNMS_TRT", "1")
    if not creator:
        raise RuntimeError("EfficientNMS_TRT plugin not found in registry")

    # Configure plugin fields with type safety
    fields = [
        ("background_class", np.array([-1], dtype=np.int32), trt.PluginFieldType.INT32),
        ("max_output_boxes", np.array([plugin.options["max_det"]], dtype=np.int32), trt.PluginFieldType.INT32),
        (
            "score_threshold",
            np.array([plugin.options["score_threshold"]], dtype=np.float32),
            trt.PluginFieldType.FLOAT32,
        ),
        (
            "iou_threshold",
            np.array([plugin.options["iou_threshold"]], dtype=np.float32),
            trt.PluginFieldType.FLOAT32,
        ),
        ("box_coding", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32),
        ("score_activation", np.array([0], dtype=np.int32), trt.PluginFieldType.INT32),
    ]

    # Create plugin field collection
    fc = trt.PluginFieldCollection([trt.PluginField(*args) for args in fields])

    # Create and add plugin to network
    trt_plugin = creator.create_plugin("nms_layer", fc)
    if not trt_plugin:
        raise RuntimeError("Failed to create EfficientNMS plugin")

    layer = network.add_plugin_v2([boxes.get_output(0), scores.get_output(0)], trt_plugin)

    boxes = layer.get_output(1)
    scores = layer.get_output(2)
    classes = layer.get_output(3)

    classes_float = network.add_identity(classes)
    classes_float.set_output_type(0, trt.float32)
    classes_reshaped = network.add_shuffle(classes_float.get_output(0))
    classes_reshaped.reshape_dims = trt.Dims([plugin.options["max_det"], 1])

    scores_reshaped = network.add_shuffle(scores)
    scores_reshaped.reshape_dims = trt.Dims([plugin.options["max_det"], 1])

    boxes_reshaped = network.add_shuffle(boxes)
    boxes_reshaped.reshape_dims = trt.Dims([plugin.options["max_det"], 4])

    concat_inputs = [boxes_reshaped.get_output(0), scores_reshaped.get_output(0), classes_reshaped.get_output(0)]
    concat_layer = network.add_concatenation(concat_inputs)
    concat_layer.axis = 1

    for i in range(network.num_layers):
        layer = network.get_layer(i)
        for j in range(layer.num_outputs):
            tensor = layer.get_output(j)
            if tensor.name == "output":
                tensor.name = "raw_output"

    output_node = concat_layer.get_output(0)

    output_reshaped = network.add_shuffle(output_node)
    output_reshaped.reshape_dims = trt.Dims([1, *output_node.shape])

    output = output_reshaped.get_output(0)
    output.name = "output"
    network.mark_output(output)

    return network


def add_batched_nms_plugin(network: trt.INetworkDefinition, plugin: Plugin) -> trt.INetworkDefinition:
    registry = trt.get_plugin_registry()
    creator = registry.get_plugin_creator("BatchedNMSDynamic_TRT", "1")
    if not creator:
        raise RuntimeError("BatchedNMSDynamic_TRT plugin not found in registry")

    # Configure plugin fields with type safety
    fields = [
        ("shareLocation", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32),
        ("backgroundLabelId", np.array([-1], dtype=np.int32), trt.PluginFieldType.INT32),
        ("numClasses", np.array([plugin.options["nc"]], dtype=np.int32), trt.PluginFieldType.INT32),
        ("topK", np.array([100], dtype=np.int32), trt.PluginFieldType.INT32),
        ("keepTopK", np.array([plugin.options["max_det"]], dtype=np.int32), trt.PluginFieldType.INT32),
        (
            "scoreThreshold",
            np.array([plugin.options["score_threshold"]], dtype=np.float32),
            trt.PluginFieldType.FLOAT32,
        ),
        (
            "iouThreshold",
            np.array([plugin.options["iou_threshold"]], dtype=np.float32),
            trt.PluginFieldType.FLOAT32,
        ),
        ("isNormalized", np.array([True], dtype=np.int32), trt.PluginFieldType.INT32),
        ("clipBoxes", np.array([False], dtype=np.int32), trt.PluginFieldType.INT32),
    ]

    # Create plugin field collection
    fc = trt.PluginFieldCollection([trt.PluginField(*args) for args in fields])

    # Create and add plugin to network
    trt_plugin = creator.create_plugin("nms_layer", fc)
    if not trt_plugin:
        raise RuntimeError("Failed to create BatchedNMSDynamic_TRT plugin")

    layer = network.add_plugin_v2([network.get_output(0), network.get_output(1)], trt_plugin)

    boxes = layer.get_output(1)
    scores = layer.get_output(2)
    classes = layer.get_output(3)

    classes_float = network.add_identity(classes)
    classes_float.set_output_type(0, trt.float32)
    classes_reshaped = network.add_shuffle(classes_float.get_output(0))
    classes_reshaped.reshape_dims = trt.Dims([-1, plugin.options["max_det"], 1])

    scores_reshaped = network.add_shuffle(scores)
    scores_reshaped.reshape_dims = trt.Dims([-1, plugin.options["max_det"], 1])

    boxes_reshaped = network.add_shuffle(boxes)
    boxes_reshaped.reshape_dims = trt.Dims([-1, plugin.options["max_det"], 4])

    concat_inputs = [boxes_reshaped.get_output(0), scores_reshaped.get_output(0), classes_reshaped.get_output(0)]
    concat_layer = network.add_concatenation(concat_inputs)
    concat_layer.axis = 2

    output_node = concat_layer.get_output(0)
    output_node.name = "output"
    network.mark_output(output_node)

    for i in range(network.num_outputs - 1, 0, -1):
        network.unmark_output(network.get_output(i - 1))

    return network


class YoloExporter(Exporter):
    def __init__(self, config: ExportConfig) -> None:
        super(YoloExporter, self).__init__(config)

    def load_checkpoints(self, weights_path: Union[str, Path], model_configuration: Optional[str] = None) -> None:
        self.model = YOLO(weights_path)
        if len(self.config.tensorrt.plugins) and self.config.onnx.specific.dynamic_axes:
            cls = WrappedModel
        else:
            cls = Model
        self.model = cls(self.model)

        device = torch.device(self.config.device)
        for p in self.model.parameters():
            p.required_grad = False
        self.model.eval()
        self.model.to(device)
        if self.config.enable_mixed_precision:
            self.model.half()
        else:
            self.model.float()
