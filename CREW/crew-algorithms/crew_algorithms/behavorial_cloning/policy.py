import torch
import torch.nn as nn
import torch.multiprocessing as mp

# from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules.models.multiagent import MultiAgentMLP


class ActorNet(nn.Module):
    def __init__(
        self, n_agent_inputs, num_cells, n_agent_outputs, activation_class
    ) -> None:
        super().__init__()
        self.mlp = MultiAgentMLP(
            n_agent_inputs=n_agent_inputs,
            num_cells=num_cells,
            n_agent_outputs=n_agent_outputs,
            centralised=False,
            n_agents=1,
            activation_class=activation_class,
            share_params=False,
        )

    def forward(self, obs: torch.Tensor):
        mlp_out = self.mlp(obs)
        print(mlp_out)
        return mlp_out


class QValueNet(nn.Module):
    def __init__(
        self, n_agent_inputs, num_cells, n_agent_outputs, activation_class
    ) -> None:
        super().__init__()
        self.mlp = MultiAgentMLP(
            n_agent_inputs=n_agent_inputs,
            num_cells=num_cells,
            n_agent_outputs=n_agent_outputs,
            centralised=False,
            n_agents=1,
            activation_class=activation_class,
            share_params=False,
        )

    def forward(self, obs: torch.Tensor):
        mlp_out = self.mlp(obs)
        return mlp_out


# class mp_model(mp.Process):
    