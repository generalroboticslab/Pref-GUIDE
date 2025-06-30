import torch
import torch.nn as nn
import torch.multiprocessing as mp

# from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules.models.multiagent import MultiAgentMLP
from crew_algorithms.auto_encoder.model import Encoder

class ActorNet(nn.Module):
    def __init__(
        self, encoder, n_agent_inputs, num_cells, n_agent_outputs, activation_class
    ) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(n_agent_inputs, num_cells),
            activation_class(),
            nn.Linear(num_cells, num_cells),
            activation_class(),
            nn.Linear(num_cells, n_agent_outputs),
        )
        self.encoder = encoder
        self.init_weights()

    def init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, obs: torch.Tensor):
        batch = False
        # print('actor', obs.shape)
        while len(obs.shape) > 3:
        # if len(obs.shape) == 5 or len(obs.shape) == 3:
            # [128, 1, 3, 84, 84], [128, 1, 9]
            obs = obs.squeeze(1)
            batch = True

        if not batch:
            with torch.inference_mode():
                obs = self.encoder(obs).flatten(1)
        else:
            obs = self.encoder(obs).flatten(1)  

        if batch:
            obs = obs.unsqueeze(1)

        # print('actor', obs.shape)
        mlp_out = self.mlp(obs)
        
        return mlp_out


class QValueNet(nn.Module):
    def __init__(
        self, encoder, n_agent_inputs, num_cells, n_agent_outputs, activation_class
    ) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(n_agent_inputs, num_cells),
            activation_class(),
            nn.Linear(num_cells, num_cells),
            activation_class(),
            nn.Linear(num_cells, n_agent_outputs),
        )

        self.encoder = encoder
        self.init_weights()

    def init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)


    def forward(self, obs: torch.Tensor):
        batch = False
        # print('qvalue', obs.shape)
        while len(obs.shape) > 3:
            obs = obs.squeeze(1)
            batch = True

        if not batch:
            with torch.inference_mode():
                obs = self.encoder(obs).flatten(1)
        else:
            obs = self.encoder(obs).flatten(1)  
            
        if batch:
            obs = obs.unsqueeze(1)
        mlp_out = self.mlp(obs)
        return mlp_out


class ContinuousActorNet(nn.Module):
    def __init__(
        self, encoder, n_agent_inputs, num_cells, out_dims, activation_class=nn.ReLU) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(n_agent_inputs, num_cells),
            nn.BatchNorm1d(num_cells),
            activation_class(),
            nn.Linear(num_cells, num_cells),
            nn.BatchNorm1d(num_cells),
            activation_class(),
            nn.Linear(num_cells, out_dims),
        )
        self.encoder = encoder
        self.init_weights()

        if isinstance(self.encoder, Encoder):
            self.num_channels = next(self.encoder.parameters()).shape[1]
        else:
            self.num_channels = None

    def init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, obs: torch.Tensor, **kwargs):
        bs = obs.shape[0]
        if bs == 1:
            self.mlp.eval()
        else:
            self.mlp.train()
        if len(obs.shape) == 5:
            obs = obs.squeeze(1)
        
        if self.num_channels is not None:
            obs = obs.view(obs.shape[0], -1, self.num_channels, obs.shape[-2], obs.shape[-1]).view(-1, self.num_channels, obs.shape[-2], obs.shape[-1])

        obs = self.encoder(obs).flatten(1).view(bs, -1)

        if 'step_count' in kwargs:
            step_count = kwargs['step_count']
            while len(step_count.shape) > 2:
                step_count = step_count.squeeze(1)
            obs = torch.cat([obs, step_count], dim=-1).to(obs.device)

        mlp_out = self.mlp(obs)
        
        return mlp_out.unsqueeze(1)
    # def forward(self, obs: torch.Tensor):
    #     bs = obs.shape[0]
    #     while len(obs.shape) >3:
    #         obs = obs.squeeze(1)

    #     # print(obs.shape)
    #     # obs = obs.flatten(1)
    #     # print(obs.shape)3

    #     obs = self.encoder(obs).flatten(1).view(bs, -1)
    #     # print(obs.shape)
    #     mlp_out = self.mlp(obs)#.clamp(-20, 20)
        
    #     # print(mlp_out)
    #     return mlp_out.unsqueeze(1)

class ContinuousQValueNet(nn.Module):
    def __init__(
        self, encoder, n_agent_inputs, num_cells, activation_class=nn.ReLU
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            # nn.Linear(n_agent_inputs + 1, num_cells),
            nn.Linear(n_agent_inputs, num_cells),
            # nn.BatchNorm1d(num_cells),
            activation_class(),
            nn.Linear(num_cells, num_cells),
            # nn.BatchNorm1d(num_cells),
            activation_class(),
            nn.Linear(num_cells, 1),
        )
        self.encoder = encoder
        self.init_weights()
        if isinstance(self.encoder, Encoder):
            self.num_channels = next(self.encoder.parameters()).shape[1]
        else:
            self.num_channels = None

    def init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, **kwargs):
        bs = obs.shape[0]
        if bs == 1:
            self.mlp.eval()
        else:
            self.mlp.train()
        if len(obs.shape) == 5:
            obs = obs.squeeze(1)
        if self.num_channels is not None:
            obs = obs.view(obs.shape[0], -1, self.num_channels, obs.shape[-2], obs.shape[-1]).view(-1, self.num_channels, obs.shape[-2], obs.shape[-1])

        obs = self.encoder(obs).flatten(1).view(bs, -1)
        while len(action.shape) > 2:
            action = action.squeeze(1)
        obs_action = torch.cat([obs, action], dim=-1).to(obs.device)

        if 'step_count' in kwargs:
            step_count = kwargs['step_count']
            while len(step_count.shape) > 2:
                step_count = step_count.squeeze(1)
            obs_action = torch.cat([obs_action, step_count], dim=-1).to(obs.device)

        mlp_out = self.mlp(obs_action)
        return mlp_out.unsqueeze(1)

    # def forward(self, obs: torch.Tensor, action: torch.Tensor):
    #     bs = obs.shape[0]
    #     while len(obs.shape) > 3:
    #         obs = obs.squeeze(1)

    #     while len(action.shape) > 2:
    #         action = action.squeeze(1)
    #     # obs = obs.flatten(1)
    #     # obs = obs.view(obs.shape[0], -1, 3, obs.shape[-2], obs.shape[-1]).view(-1, 3, obs.shape[-2], obs.shape[-1])

    #     obs = self.encoder(obs).flatten(1).view(bs, -1)

    #     # print(obs.shape, action.shape) 
    #     mlp_out = self.mlp(torch.cat([obs, action], dim=-1))#.clamp(-20, 20)

    #     return mlp_out.unsqueeze(1)


class FeedbackNet(nn.Module):
    def __init__(
        self, encoder, n_agent_inputs, num_cells, activation_class=nn.ReLU
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_agent_inputs, num_cells),
            # nn.BatchNorm1d(num_cells),
            activation_class(),
            nn.Linear(num_cells, num_cells),
            # nn.BatchNorm1d(num_cells),
            activation_class(),
            nn.Linear(num_cells, 1),
        )
        self.encoder = encoder
        self.init_weights()
        if isinstance(self.encoder, Encoder):
            self.num_channels = next(self.encoder.parameters()).shape[1]
        else:
            self.num_channels = None

    def init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        if bs == 1:
            self.mlp.eval()
        else:
            self.mlp.train()
        bs = obs.shape[0]
        if len(obs.shape) == 5:
            obs = obs.squeeze(1)
        obs = obs.view(obs.shape[0], -1, self.num_channels, obs.shape[-2], obs.shape[-1]).view(-1, 3, obs.shape[-2], obs.shape[-1])

        obs = self.encoder(obs).flatten(1).view(bs, -1)
        action = action.view(bs, -1)
        # feedback = feedback.view(bs, -1)

        # while len(action.shape) > 3:
        #     action = action.squeeze(1)
        # while len(feedback.shape) > 2:
        #     feedback = feedback.squeeze(1)

        obs_action = torch.cat([obs, action], dim=-1).to(obs.device)
        mlp_out = self.mlp(obs_action)

        return mlp_out.unsqueeze(1)


class PbNet(nn.Module):
    """ Preference-based network. Given obs and action, predict assigned feedback.
    Args:
    """
    def __init__(
        self, encoder, n_agent_inputs, num_cells, activation_class=nn.ReLU, use_activation=False
    ) -> None:
        super().__init__()
        
        last_layer = nn.Tanh() if use_activation else nn.Identity()
        
        self.mlp = nn.Sequential(
            # nn.Linear(n_agent_inputs + 1, num_cells),
            nn.Linear(n_agent_inputs, num_cells),
            # nn.BatchNorm1d(num_cells),
            activation_class(),
            nn.Linear(num_cells, num_cells),
            # nn.BatchNorm1d(num_cells),
            activation_class(),
            nn.Linear(num_cells, 1),
            last_layer
        )
        self.encoder = encoder
        self.init_weights()
        
        if isinstance(self.encoder, Encoder):
            self.num_channels = next(self.encoder.parameters()).shape[1]
        else:
            self.num_channels = None

    def init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                m.bias.data.fill_(0.01)
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor, **kwargs):
        bs = obs.shape[0]
        if bs == 1:
            self.mlp.eval()
        else:
            self.mlp.train()
        
        if len(obs.shape) == 5:
            obs = obs.squeeze(1)
        if self.num_channels is not None:
            try:
                obs = obs.view(obs.shape[0], -1, self.num_channels, obs.shape[-2], obs.shape[-1]).view(-1, self.num_channels, obs.shape[-2], obs.shape[-1])
            except:
                obs = obs.reshape(obs.shape[0], -1, self.num_channels, obs.shape[-2], obs.shape[-1]).reshape(-1, self.num_channels, obs.shape[-2], obs.shape[-1])
            
        obs = self.encoder(obs).flatten(1).view(bs, -1)
        # print(obs.shape)
        while len(action.shape) > 2:
            action = action.squeeze(1)
        obs_action = torch.cat([obs, action], dim=-1).to(obs.device)

        if 'step_count' in kwargs:
            step_count = kwargs['step_count']
            while len(step_count.shape) > 2:
                step_count = step_count.squeeze(1)
            obs_action = torch.cat([obs_action, step_count], dim=-1).to(obs.device)

        mlp_out = self.mlp(obs_action)
        return mlp_out.unsqueeze(1)
    