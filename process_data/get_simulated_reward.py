from dataloader import FeedbackDataset
from models import CNN_MLP_Fusion
from torch.utils.data import DataLoader
import torch
import numpy as np
import random
from utils import plot_comparison, plot_feedback
from config import get_config
from loss import count_ranking

config = get_config()

#Get the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed
torch.manual_seed(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)

top_15_list = [1,2,7,10,13,15,17,18,25,30,32,33,37,39,44]
# read_tensor
no_preference = "_no_preference"
game_name = "hide_and_seek_1v1" #config.game

for id in top_15_list:
    
    id_list = top_15_list.copy()
    id_list.remove(id)
    
    if id < 10:
        id = f"0{id}"
    image_tensor = torch.load(f"../Preference_guide_Data/{game_name}/{id}/{id}_img_recovered.pt")
    action_tensor = torch.load(f"../Preference_guide_Data/{game_name}/{id}/{id}_action.pt")
    model_list = []

    

    for ref_id in id_list:
        
        weights = f"reward_model/{game_name}_subject_{ref_id}_pair_2-reference_model_vote_ref_activation.pt"

        model = CNN_MLP_Fusion(num_frames=config.traj_length*3)
        model.load_state_dict(torch.load(weights))
        model = model.to(device)
        model_list.append(model)

    predicted_reward_list = []
    image_tensor = image_tensor.to(device)
    action_tensor = action_tensor.to(device).squeeze(-2)

    for model in model_list:

        # #calculate the accuracy of prediction 
        with torch.no_grad():
            model.eval()

            simulated_reward = model(image_tensor,action_tensor).squeeze().cpu()
            predicted_reward_list.append(simulated_reward.unsqueeze(0))
            

    reward = torch.cat(predicted_reward_list, dim=0)
    print(reward.shape)

    torch.save(reward, f"../Preference_guide_Data/{game_name}/{id}/{id}_simulated_reward_recovered_soft_label.pt")