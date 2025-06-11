import torch
from typing import List


class WrappedModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super(WrappedModel, self).__init__()
        self.nc = model.nc
        self.model: torch.nn.Module = model.model

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        predictions = self.model(x)
        prediction = predictions[0] if isinstance(predictions, tuple) else predictions
        prediction = prediction.transpose(2, 1)
        boxes, scores = prediction.split([4, self.model.nc], dim=2)

        x, y, w, h = boxes.unbind(dim=-1)
        boxes = torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dim=-1)
        bs, anchors, xyxy = boxes.shape

        return [boxes.reshape(bs, anchors, 1, xyxy), scores]

class Model(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super(Model, self).__init__()
        self.nc = model.nc
        self.model: torch.nn.Module = model.model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        return outputs[0] if isinstance(outputs, (tuple, list)) else outputs
