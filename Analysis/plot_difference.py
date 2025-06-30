import torch
import random
import numpy as np
from utils_plot_difference import ChannelReducer, StateActionEmbedder, reduce_to_2D, plot_position_feedback
from timm import create_model
import os 

# Fix all seeds
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

top_15_list = list(range(52))
top_15_list.remove(12)
top_15_list.remove(42)
top_15_list = [0]

game_name_list = ["hide_and_seek_1v1"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = "human"

dino = create_model('vit_small_patch16_224_dino', pretrained=True).to(device)
dino.eval()


for game_name in game_name_list:
    
    reducer = ChannelReducer(in_channel=1 if game_name=="bowling" else 9).to(device)
    embedder = StateActionEmbedder(dino_model=dino, reducer=reducer).to(device)
    
    for id in top_15_list:
        print(id)
        if int(id) < 10:
            id = "0" + str(id)

        image_tensor = torch.load(f"../../Preference_guide_Data/{game_name}/{id}/{id}_img.pt")
        action_tensor = torch.load(f"../../Preference_guide_Data/{game_name}/{id}/{id}_action.pt")
        human_feedback_tensor = torch.load(f"../../Preference_guide_Data/{game_name}/{id}/{id}_{data}_feedback.pt")

        # Compute embeddings
        image_tensor = image_tensor.squeeze(1).to(device)
        action_tensor = action_tensor.squeeze().to(device)
        human_feedback_tensor = human_feedback_tensor.squeeze().to(device)
        
        human_feedback_tensor = (human_feedback_tensor - human_feedback_tensor.min())/(human_feedback_tensor.max()-human_feedback_tensor.min()) * 2 -1

        with torch.inference_mode():
            embeddings = embedder(image_tensor, action_tensor)  # [B, D+2]
        position, feedback = reduce_to_2D(embeddings,human_feedback_tensor)

        if not os.path.exists(f"embedding_vs_feedback/{game_name}"):
            os.makedirs(f"embedding_vs_feedback/{game_name}")
        
        plot_position_feedback(position,feedback,f"embedding_vs_feedback/{game_name}/subject_{id}_{data}")