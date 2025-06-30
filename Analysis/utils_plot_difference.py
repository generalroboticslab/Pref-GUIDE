import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


# Channel reduction layer: 9 → 3 channels
class ChannelReducer(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.reduce = nn.Conv2d(in_channel, 3, kernel_size=1)

    def forward(self, x):
        return self.reduce(x)

# Full embedding model
class StateActionEmbedder(nn.Module):
    def __init__(self, dino_model, reducer, embed_dim=128, action_dim=2):
        super().__init__()
        self.reducer = reducer
        self.dino = dino_model
        self.embed_dim = embed_dim
        self.action_dim = action_dim

    def forward(self, state_batch, action_batch):
        # state_batch: [B, 9, 100, 100]
        x = self.reducer(state_batch)  # [B, 3, 100, 100]
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = (x - 0.5) / 0.5  # Normalize to [-1, 1] as expected by DINO
        with torch.no_grad():
            state_embed = self.dino.forward_features(x)  # [B, embed_dim]
        state_action_embed = torch.cat([state_embed.view(state_embed.shape[0],-1), action_batch], dim=1)  # [B, embed_dim + 2]
        return state_action_embed

def reduce_to_2D(embeddings,feedbacks):
    emb_np = embeddings.detach().cpu().numpy()

# Apply t-SNE (reduce to 2D)
    tsne = TSNE(n_components=2, perplexity=5, init='pca', random_state=0)
    emb_2d = tsne.fit_transform(emb_np)  # [N, 2]

    # Optional: color by feedback or cluster ID
    feedback_np = feedbacks.detach().cpu().numpy()
    return emb_2d, feedback_np

import matplotlib.pyplot as plt
import numpy as np

def plot_position_feedback(position: np.ndarray, feedback: np.ndarray, title: str = "Position vs Feedback"):
    """
    Plot 2D positions with color-coded feedback values.

    Args:
        position (np.ndarray): Shape [N, 2], 2D positions.
        feedback (np.ndarray): Shape [N], feedback values for coloring.
        title (str): Title of the plot.
    """
    assert position.shape[1] == 2, "Position must be 2D"
    assert len(position) == len(feedback), "Position and feedback must have same length"

    plt.figure(figsize=(20, 20))
    ax = plt.gca()  
    # Setting Background colour yellow 
    ax.set_facecolor("#EAEAF2")
    scatter = plt.scatter(position[:, 0], position[:, 1], c=feedback, cmap='viridis', s=500, alpha=0.8)
    cbar = plt.colorbar(scatter, ticks=[-1, -0.5, 0.0, 0.5, 1])
    
    # cbar.set_label('Human Feedback', fontsize=24)
    cbar.ax.tick_params(labelsize=48)
    plt.xticks([])
    plt.yticks([])

    # ax.grid(True, color='white', linewidth=4)
    # Make axis lines (spines) thicker
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    plt.tight_layout()
    plt.savefig(f"{title}.png")