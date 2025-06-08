import torch


class WrappedModel(torch.nn.Module):
    def __init__(self, original_model: torch.nn.Module):
        super().__init__()
        self.model: torch.nn.Module = original_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        return outputs[0] if isinstance(outputs, (tuple, list)) else outputs

