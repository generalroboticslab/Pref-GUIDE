from utils import train_model, load_data, make_model, eval_model
import torch
import numpy as np
import random
from config import get_config

config = get_config()
config.pretrain = bool(config.pretrain)
config.use_activation = bool(config.use_activation)
config.no_preference_window = bool(config.no_preference_window)
config.moving_window = bool(config.moving_window)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

trainLoader, validationLoader, train_index,_ = load_data(config)

model, optimizer, criterion = make_model(config, device)

#training loop
best_model = train_model(model, config, trainLoader, validationLoader, device, optimizer, criterion)

model.load_state_dict(best_model)
eval_model(model, config, train_index, device)
