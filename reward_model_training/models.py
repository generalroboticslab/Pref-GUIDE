import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from PIL import Image
import random
class Encoder(nn.Module):
    def __init__(self, in_channels: int=3, embedding_dim: int = 128, eval=False):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=8, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(576, embedding_dim),
        )
        
        self.evaluation_mode = eval


    def forward(self, x):
        # print(x.shape)
        if len(x.shape) < 4:
            x = x.unsqueeze(1)
        if x.shape[0] > 1 and not self.evaluation_mode:
            # print("Encoder has augment")
            pix = 8
            a, b = random.uniform(-pix, pix), random.uniform(-pix, pix)
            x = v2.functional.affine(x, angle=0, translate=[a, b], scale=1, shear=0)
        else:
            # print("Encoder in Evaluation mode")
            self.cnn.eval()
            self.fc.eval()

        x = self.cnn(x)
        x = self.fc(x)
        return x

class Feedback_Net(nn.Module):
    def __init__(
        self, encoder, n_agent_inputs, num_cells, activation_class=nn.ReLU, eval=False, use_activation=False,
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

        self.evaluation_mode = eval
        
    def init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                m.bias.data.fill_(0.01)
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor, **kwargs):
        bs = obs.shape[0]
        # print(f"Input shape: {obs.shape}, Action shape: {action.shape}, Batch size: {bs}")
        if bs == 1:
            # print("MLP in Evaluation mode")
            self.mlp.eval()
        else:
            if not self.evaluation_mode:
                # print("MLP in training")
                self.mlp.train()
            else:
                # print("MLP in Evaluation mode")
                self.mlp.eval()
                
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