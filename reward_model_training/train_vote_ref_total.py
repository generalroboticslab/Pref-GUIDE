from utils import train_model, load_data_vote_ref_total, make_model, eval_model
import torch
import numpy as np
import random
from config import get_config

config = get_config()
config.pretrain = bool(config.pretrain)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

trainLoader, validationLoader = load_data_vote_ref_total(config)

# model, optimizer, criterion = make_model(config, device)

# #training loop
# best_model = train_model(model, config, trainLoader, validationLoader, device, optimizer, criterion)

# model.load_state_dict(best_model)
# eval_model(model, config, train_index, device)
