import torch 
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from models import Feedback_Net, Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_feedback(human_feedback,heuristic_feedback,simulated_feedback, name):
    human_feedback = np.array(human_feedback)
    heuristic_feedback = np.array(heuristic_feedback)
    simulated_feedback = np.array(simulated_feedback)
    
    plt.figure(figsize=(20, 5))  # Set figure size
    plt.plot(human_feedback,linewidth=3)
    plt.plot(heuristic_feedback,linewidth=3)
    plt.plot(simulated_feedback,linewidth=2, linestyle='--', color='purple')
    plt.legend(['Human Feedback', 'Heuristic Feedback', "GUIDE"], fontsize=10)
    
    plt.ylabel('Feedback')
    plt.xlabel('Step')
    plt.savefig(f'{name}.png', bbox_inches='tight')  # Save with tight layout
    plt.title("Human Feedback")
    plt.close()  # Close the figure to free memory
    
id = 18
eval_model = "vote_ref_pretrain_activation_by_traj"
game_name = "hide_and_seek_1v1"

image_tensor = torch.load(f"../../Preference_guide_Data/{game_name}/{id}/{id}_img.pt")
action_tensor = torch.load(f"../../Preference_guide_Data/{game_name}/{id}/{id}_action.pt")
human_feedback_tensor = torch.load(f"../../Preference_guide_Data/{game_name}/{id}/{id}_human_feedback.pt")
heuristic_feedback_tensor = torch.load(f"../../Preference_guide_Data/{game_name}/{id}/{id}_heuristic_feedback.pt")
traj_ids = torch.load(f"../../Preference_guide_Data/{game_name}/{id}/{id}_tarj_ids.pt")
validation_traj = traj_ids % 10 == 9

model = Feedback_Net(
    encoder=Encoder(in_channels=3, embedding_dim=64, eval=True),
    n_agent_inputs=2 + 64*3,
    num_cells=256,
    activation_class=torch.nn.ReLU,
    eval=True,
    use_activation = eval_model.find("activation") != -1
)

model_weights = f"../reward_model/{eval_model}/{game_name}/subject_{18}.pt"
model.load_state_dict(torch.load(model_weights))
model = model.to(device)
model.eval()

image_tensor = image_tensor.to(device)
action_tensor = action_tensor.to(device)
with torch.inference_mode():
    # Get simulated feedback from the model
    simulated_feedback = model(image_tensor, action_tensor)



human_feedback_tensor = (human_feedback_tensor - human_feedback_tensor.min())/(human_feedback_tensor.max() - human_feedback_tensor.min()) * 2 - 1
heuristic_feedback_tensor = (heuristic_feedback_tensor - heuristic_feedback_tensor.min())/(heuristic_feedback_tensor.max() - heuristic_feedback_tensor.min()) * 2 - 1




start_index = 600
end_index = 800

plot_feedback(
    human_feedback_tensor[validation_traj].squeeze(),
    heuristic_feedback_tensor[validation_traj].squeeze(),
    simulated_feedback[validation_traj].squeeze().detach().cpu().numpy(),
    eval_model)

