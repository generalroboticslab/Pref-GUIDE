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
num_stacks = 3
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
    image_tensor = torch.load(f"../Preference_guide_Data/{game_name}/{id}/{id}_img.pt")
    action_tensor = torch.load(f"../Preference_guide_Data/{game_name}/{id}/{id}_action.pt")
    
    
    stack_obs_tensor = torch.zeros((image_tensor.shape[0], 1, num_stacks*3, 100, 100))
    satck_action_tensor = torch.zeros((image_tensor.shape[0], 1,1, num_stacks*2))
    
    for i in range(image_tensor.shape[0]):
        for j in range(num_stacks):
            if i-j >= 0:
                stack_obs_tensor[i,0,j*3:(j+1)*3,:,:] = image_tensor[i-j][:,-3:,:,:]
                satck_action_tensor[i,0,0,j*2:(j+1)*2] = action_tensor[i-j]
            else:
                stack_obs_tensor[i,0,j*3:(j+1)*3,:,:] = image_tensor[0][:,-3:,:,:]
                satck_action_tensor[i,0,0,j*2:(j+1)*2] = action_tensor[0]
    model_list = []

    

    for ref_id in id_list:
        
        weights = f"regression_model/{game_name}_subject_{ref_id}_regression_model_{num_stacks}_stacks.pt"

        model = CNN_MLP_Fusion(num_frames=config.traj_length*num_stacks, use_activation=False)
        model.load_state_dict(torch.load(weights))
        model = model.to(device)
        model_list.append(model)

    predicted_reward_list = []
    image_tensor = stack_obs_tensor.to(device)
    action_tensor = satck_action_tensor.to(device).squeeze(-2)
    # image_tensor = image_tensor.to(device)
    # action_tensor = action_tensor.to(device).squeeze(-2)

    for model in model_list:

        # #calculate the accuracy of prediction 
        with torch.no_grad():
            model.eval()

            simulated_reward = model(image_tensor,action_tensor).squeeze().cpu()
            predicted_reward_list.append(simulated_reward.unsqueeze(0))
            

    reward = torch.cat(predicted_reward_list, dim=0)
    print(reward.shape)

    torch.save(reward, f"../Preference_guide_Data/{game_name}/{id}/{id}_simulated_reward_regression.pt")