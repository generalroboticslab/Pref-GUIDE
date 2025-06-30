import torch 
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from models import Feedback_Net, Encoder
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using CUDA
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

subject_list = [1,2,7,10,13,15,17,18,25,30,32,33,37,39,44]
subject_list = list(range(52))
subject_list.remove(12)
subject_list.remove(42)

eval_model_list = ["guide_model", "hard_label_pretrain_activation"]  # Options: "hard_label", "soft_label_1e-3", "vote_ref_soft_label_1e-3", "regression"
game_name = "find_treasure"  # Options: "find_treasure", "bowling", "hide_and_seek_1v1"

count = 0
human_difference = 0
heuristic_difference = 0

for eval_model in eval_model_list:
    for eval_id in subject_list:
        
        model = Feedback_Net(
            encoder=Encoder(in_channels=3, embedding_dim=64, eval=True),
            n_agent_inputs=2 + 64*3,
            num_cells=256,
            activation_class=torch.nn.ReLU,
            eval=True, 
            use_activation = eval_model.find("activation") != -1
        )
        model_weights = f"../reward_model/{eval_model}/{game_name}/subject_{eval_id}.pt"
        model.load_state_dict(torch.load(model_weights))
        model = model.to(device)
        model.eval()
        
        for id in subject_list:
            if int(id) != int(eval_id):
                continue
            if int(id) < 10:
                id = "0" + str(id)
            image_tensor = torch.load(f"../../Preference_guide_Data/{game_name}/{id}/{id}_img.pt")
            action_tensor = torch.load(f"../../Preference_guide_Data/{game_name}/{id}/{id}_action.pt")
            human_feedback_tensor = torch.load(f"../../Preference_guide_Data/{game_name}/{id}/{id}_human_feedback.pt")
            heuristic_feedback_tensor = torch.load(f"../../Preference_guide_Data/{game_name}/{id}/{id}_heuristic_feedback.pt")
            traj_ids = torch.load(f"../../Preference_guide_Data/{game_name}/{id}/{id}_tarj_ids.pt")
            
            validation_traj = traj_ids % 10 == 9
            # index = torch.arange(0,len(image_tensor))[validation_traj]

            image_tensor = image_tensor[validation_traj].to(device)
            action_tensor = action_tensor[validation_traj].to(device)
            with torch.inference_mode():
                # Get simulated feedback from the model
                simulated_feedback = model(image_tensor, action_tensor)
            simulated_feedback = simulated_feedback.squeeze().cpu()
            
            human_feedback_tensor = (human_feedback_tensor - human_feedback_tensor.min())/(human_feedback_tensor.max() - human_feedback_tensor.min()) * 2 - 1
            heuristic_feedback_tensor = (heuristic_feedback_tensor - heuristic_feedback_tensor.min())/(heuristic_feedback_tensor.max() - heuristic_feedback_tensor.min()) * 2 - 1
            
            human_feedback_tensor = human_feedback_tensor[validation_traj]
            heuristic_feedback_tensor = heuristic_feedback_tensor[validation_traj]
            
            assert len(human_feedback_tensor) == len(simulated_feedback)
            assert len(simulated_feedback) == len(heuristic_feedback_tensor)
            
            # human_difference += torch.mean((simulated_feedback- human_feedback_tensor)**2).item()
            # heuristic_difference += torch.mean((simulated_feedback - heuristic_feedback_tensor)**2).item()
            
            human_difference += torch.mean(torch.abs(simulated_feedback- human_feedback_tensor)).item()
            heuristic_difference += torch.mean(torch.abs(simulated_feedback - heuristic_feedback_tensor)).item()
            
            count += 1
            
            # print(f"Subject {eval_id} count {count}")

    print("eval_model:", eval_model, "human_difference:", human_difference/count, "heuristic_difference:", heuristic_difference/count)

