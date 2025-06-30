import torch 
import numpy as np
import matplotlib.pyplot as plt
from models import Feedback_Net, Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Feedback_Net(
    encoder=Encoder(in_channels=3, embedding_dim=64, eval=True),
    n_agent_inputs=2 + 64*3,
    num_cells=256,
    activation_class=torch.nn.ReLU,
    eval=True
)
subject_id = 1
layer_key = "encoder.cnn.7.weight"

model_weights = torch.load(f"../reward_model_top_15/regression_freeze/hide_and_seek_1v1/subject_{subject_id}.pt")
pre_train_weights = torch.load(f"../reward_model/guide_model_RL/hide_and_seek_1v1/subject_{subject_id}.pt")
print((pre_train_weights[f"1.module.{layer_key}"] == model_weights[layer_key]).prod())
