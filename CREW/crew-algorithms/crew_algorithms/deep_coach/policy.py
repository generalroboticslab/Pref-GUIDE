import torch
import torch.nn as nn
from torchrl.modules import MLP


class PolicyNet(nn.Module):
    def __init__(self, num_actions: int) -> None:
        """The policy network specified by the DeepCoach paper.

        Args:
            num_actions: The number of possible actions the network
                should output.
        """
        super().__init__()
        self.mlp = MLP(
            in_features=128,
            out_features=num_actions,
            num_cells=[30, 30],
            activation_class=nn.ReLU,
            activate_last_layer=False,
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, obs: torch.Tensor):
        mlp_out = self.mlp(obs)
        softmax_out = self.softmax(mlp_out)
        return softmax_out
