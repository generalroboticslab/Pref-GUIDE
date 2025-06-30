import torch
import torch.nn as nn
from torchrl.data.tensor_specs import TensorSpec


class PolicyNet(nn.Module):
    def __init__(self, action_spec: TensorSpec):
        super().__init__()
        self.action_spec = action_spec

    def forward(self, obs: torch.Tensor):
        return self.action_spec.rand()
