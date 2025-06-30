from dataloader import FeedbackDataset
from models import Feedback_Net, Encoder
import torch
import numpy as np
import random
from config import get_config

#Get the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# top_15_list = [1,2,7,10,13,15,17,18,25,30,32,33,37,39,44]
top_15_list = list(range(52))
top_15_list.remove(12)
top_15_list.remove(42)

game_name = "find_treasure"
ref_model_list = [
    "hard_label_pretrain_activation",
]

for ref_model in ref_model_list:
    for id in top_15_list:
        
        if id < 10:
            id = f"0{id}"
        image_tensor = torch.load(f"../../Preference_guide_Data/{game_name}/{id}/{id}_img.pt")
        action_tensor = torch.load(f"../../Preference_guide_Data/{game_name}/{id}/{id}_action.pt")
        model_list = []

        

        for ref_id in top_15_list:
            
            weights = f"../reward_model/{ref_model}/{game_name}/subject_{ref_id}.pt"

            encoder = Encoder(in_channels=3, embedding_dim=64, eval=True)
            
            use_activation = weights.find("activation") != -1
            
            print("use_activation:",use_activation)
            
            model = Feedback_Net(encoder=encoder, n_agent_inputs=2 + 64*3, num_cells=256, activation_class=torch.nn.ReLU, eval=True, use_activation=use_activation)
            # breakpoint()
            model.load_state_dict(torch.load(weights))
            model = model.to(device)
            model.eval()
            model_list.append(model)

        predicted_reward_list = []
        image_tensor = image_tensor.to(device)
        action_tensor = action_tensor.to(device)

        for model in model_list:

            # #calculate the accuracy of prediction 
            with torch.inference_mode():
            
                simulated_reward = model(image_tensor,action_tensor).squeeze().cpu()
                predicted_reward_list.append(simulated_reward.unsqueeze(0))

        reward = torch.cat(predicted_reward_list, dim=0)
        print(reward.shape)

        torch.save(reward, f"../../Preference_guide_Data/{game_name}/{id}/{id}_simulated_reward_{ref_model}.pt")