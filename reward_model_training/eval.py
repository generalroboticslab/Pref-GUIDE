from dataloader import FeedbackDataset
from models import CNN_MLP_Fusion
from torch.utils.data import DataLoader
import torch
import numpy as np
import random
from utils import plot_comparison, plot_feedback
from config import get_config
from itertools import combinations

id_list = [1,2,7,10,13,15,17,18,25,30,32,33,37,39,44]
num_stacks = 4
str_id_list = [str(i) if i >= 10 else "0" + str(i) for i in id_list]
 
config = get_config()

#Get the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed
torch.manual_seed(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)

total_difference = 0
game_name = "hide_and_seek_1v1" #config.game
# read_tensor
for model_id in id_list:
    for id in str_id_list:
        if int(id) != model_id:
            continue
        
        image_tensor = torch.load(f"../Preference_guide_Data/{game_name}/{id}/{id}_img.pt")
        action_tensor = torch.load(f"../Preference_guide_Data/{game_name}/{id}/{id}_action.pt")
        heuristic_feedback_tensor = torch.load(f"../Preference_guide_Data/{game_name}/{id}/{id}_heuristic_feedback.pt")
        human_feedback_tensor = torch.load(f"../Preference_guide_Data/{game_name}/{id}/{id}_human_feedback.pt")
        
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

        # model_name = f"regression_reward/{game_name}_subject_{model_id}_pair_2-reference_model_vote_ref_activation_no_preference_level_one.pt"
        model_name = f"regression_model/{game_name}_subject_{model_id}_regression_model_{num_stacks}_stacks.pt"
        model = CNN_MLP_Fusion(num_frames=config.traj_length*num_stacks)

        model.load_state_dict(torch.load(f"{model_name}"))
        model = model.to(device)
        model.eval()
        
            #calculate the accuracy of prediction 
        with torch.no_grad():
            predicted_feedback = model(stack_obs_tensor.to(device), satck_action_tensor.to(device)).squeeze().cpu()
            
            
            
            
            
            # predicted_feedback = (predicted_feedback - torch.min(predicted_feedback)) / (torch.max(predicted_feedback) - torch.min(predicted_feedback)) *2 -1

            
            #normalized the feedback 
        heuristic_feedback_tensor = (heuristic_feedback_tensor - torch.min(heuristic_feedback_tensor)) / (torch.max(heuristic_feedback_tensor) - torch.min(heuristic_feedback_tensor)) *2 -1
        human_feedback_tensor = (human_feedback_tensor - torch.min(human_feedback_tensor)) / (torch.max(human_feedback_tensor) - torch.min(human_feedback_tensor)).squeeze().cpu() *2 -1
        
        heuristic_feedback_tensor = heuristic_feedback_tensor.squeeze()
        human_feedback_tensor = human_feedback_tensor.squeeze()

        total_difference += torch.mean(torch.abs(predicted_feedback - human_feedback_tensor)).item()
        
        import matplotlib.pyplot as plt

        plt.figure(figsize=(20, 5))
        plt.plot(human_feedback_tensor, label='Real Feedback')
        plt.plot(predicted_feedback, label='Predicted Feedback')
        plt.legend()
        # plt.savefig(f"{model_name}_eval_{id}.png")

print("Average difference: ", total_difference/15)