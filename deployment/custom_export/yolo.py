import cv2
import os.path
import numpy as np
from pathlib import Path
from mmcv.visualization.image import imshow_det_bboxes
import torch
import tensorrt as trt
from typing import Union, Tuple
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_boxes, non_max_suppression

from deployment.core.executors.factory import ExtendExecutor
from deployment.models.export import ExportConfig
from deployment.core.exporters.factory import Exporter
from utils.logger import get_project_root


class YoloExecutor(ExtendExecutor):
    def __init__(self, config: ExportConfig) -> None:
        super(YoloExecutor, self).__init__(config)
        self.letterbox = LetterBox(new_shape=self.config.input_shape)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush'
        ]

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocessed = torch.from_numpy(self.letterbox(image=image)).to(self.config.device)
        preprocessed = preprocessed.permute(2, 0, 1)
        preprocessed = preprocessed / 255.0
        return preprocessed[None]

    def postprocess(self, output: torch.Tensor, orig_shape) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if output.dim() == 3:
            detections = non_max_suppression(output)[0]
            boxes = scale_boxes(self.config.input_shape, detections[:, :4], orig_shape).cpu().numpy()
            scores = detections[:, 4:5].reshape(-1, 1).cpu().numpy()
            classes = detections[:, 5:].reshape(-1).to(torch.int).cpu().numpy()
        else:
            scores = output[:, 4:5].reshape(-1).cpu().numpy()
            boxes = output[:, :4][scores > 0.1]
            boxes = scale_boxes(self.config.input_shape, boxes, orig_shape).cpu().numpy()
            classes = output[:, 5:][scores > 0.1].to(torch.int32).reshape(-1).cpu().numpy()
            scores = scores[scores > 0.1].reshape(-1, 1)
        return boxes, scores, classes

    def plotter(self, backend) -> None:
        file_path = f"{get_project_root()}/deployment/resources/demo.jpg"
        if not os.path.exists(file_path):
            self.logger.warning(f"Demo file is not exist: {file_path}, skip visualization step")
            return

        image = cv2.imread(file_path)
        tensor = self.preprocess(image)
        output = self.infer(tensor, asynchronous=True)[0]
        boxes, scores, classes = self.postprocess(output, image.shape[:2])
        imshow_det_bboxes(image, np.concat([boxes, scores], axis=1), classes, self.class_names,
                          bbox_color=(0, 233, 255), text_color=(0, 233, 255), thickness=2, show=True,
                          win_name=backend)


class WrappedModel(torch.nn.Module):
    def __init__(self, original_model: torch.nn.Module):
        super().__init__()
        self.model: torch.nn.Module = original_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        return outputs[0] if isinstance(outputs, (tuple, list)) else outputs


class YoloExporter(Exporter):
    def __init__(self, config: ExportConfig):
        super(YoloExporter, self).__init__(config)

    def load_checkpoints(self, weights_path: Union[str, Path]) -> None:
        self.model = YOLO(weights_path)
        self.model = WrappedModel(self.model.model)

    def register_tensorrt_plugins(self, network: trt.INetworkDefinition) -> trt.INetworkDefinition:
        """Register and configure TensorRT plugins for YOLO model with EfficientNMS.

        Args:
            network: TensorRT network definition to modify

        Returns:
            Modified network with plugins integrated
        """

        plugin = next((plugin for plugin in self.config.tensorrt_opts.plugins if plugin.name == "efficient_nms"), None)
        if not plugin:
            return network

        previous_output = network.get_output(0)
        network.unmark_output(previous_output)

        # Prepare tensors with better dimension handling
        # Transpose and reshape operations
        shuffle_layer = network.add_shuffle(previous_output)
        shuffle_layer.second_transpose = (0, 2, 1)

        # Get dimensions with error checking
        try:
            bs, num_boxes, temp = shuffle_layer.get_output(0).shape
        except Exception as e:
            raise ValueError(f"Invalid output shape: {str(e)}")

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
            ("score_threshold", np.array([plugin.options["score_threshold"]], dtype=np.float32), trt.PluginFieldType.FLOAT32),
            ("iou_threshold", np.array([plugin.options["iou_threshold"]], dtype=np.float32), trt.PluginFieldType.FLOAT32),
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

        concat_inputs = [
            boxes_reshaped.get_output(0),
            scores_reshaped.get_output(0),
            classes_reshaped.get_output(0)
        ]
        concat_layer = network.add_concatenation(concat_inputs)
        concat_layer.axis = 1

        for i in range(network.num_layers):
            layer = network.get_layer(i)
            for j in range(layer.num_outputs):
                tensor = layer.get_output(j)
                if tensor.name == "output":
                    tensor.name = "raw_output"

        output_node = concat_layer.get_output(0)
        output_node.name = "output"
        network.mark_output(output_node)

        return network
