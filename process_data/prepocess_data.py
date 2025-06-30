from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
# from utils import plot_feedback, save_image1
import torch
import os

human_subject_id_list = [1]

S = 20000

game_name = "hide_and_seek_1v1"

for human_subject_id in human_subject_id_list:
    human_subject_id = str(human_subject_id) if human_subject_id >= 10 else "0" + str(human_subject_id)
    print(human_subject_id)
    storage = LazyMemmapStorage(S)

    rb = TensorDictReplayBuffer(storage = storage)
    try:
        rb.loads(f"guide_data/{human_subject_id}_Data/saved_training/{human_subject_id}_{game_name}_guide/prb.pkl")
    except:
        rb.loads(f"guide_data/{human_subject_id}_data/saved_training/{human_subject_id}_{game_name}_guide/prb.pkl")
    data = rb.sample(S)

    img = data["agents","observation","obs_0_0"]
    index = data["index"]
    human_feedback = data["next","agents","feedback"]
    if game_name != "bowling":
        heuristic_feedback = data["next","agents","heuristic_feedback"]
    action = data["agents","action"]
    done = data["agents","done"]

    sorted_index = index[index.argsort()].unique()

    data_length = len(sorted_index)
    sorted_img = torch.zeros((data_length,img.shape[1],img.shape[2],img.shape[3],img.shape[4]))
    sorted_action = torch.zeros((data_length,action.shape[1],action.shape[2],action.shape[3]))
    sorted_done = torch.zeros((data_length,1,1))
    sorted_human_feedback = torch.zeros((data_length,1,1))
    sorted_heuristic_feedback = torch.zeros((data_length,1,1))
    
    print(sorted_human_feedback.sum())

    index_list = []
    new_index_list = []
    for i in range(len(index.argsort())):
        if index[index.argsort()[i]].item() in new_index_list:
            continue
        new_index_list.append(index[index.argsort()[i]].item())
        index_list.append(index.argsort()[i].item())

    for i,ind in enumerate(index_list):
        # print(i,ind)
        sorted_img[i] = img[ind]
        sorted_action[i] = action[ind]
        sorted_human_feedback[i] = human_feedback[ind]
        if game_name != "bowling":
            sorted_heuristic_feedback[i] = heuristic_feedback[ind]

    print(sorted_img.shape)

    # if not os.path.exists(f"Preference_guide_Data/{game_name}/{human_subject_id}"):
    #     os.makedirs(f"Preference_guide_Data/{game_name}/{human_subject_id}")

    # torch.save(sorted_img,f"Preference_guide_Data/{game_name}/{human_subject_id}/{human_subject_id}_img.pt")
    # torch.save(sorted_action,f"Preference_guide_Data/{game_name}/{human_subject_id}/{human_subject_id}_action.pt")
    # torch.save(sorted_human_feedback,f"Preference_guide_Data/{game_name}/{human_subject_id}/{human_subject_id}_human_feedback.pt")
    # if game_name != "bowling":
    #     torch.save(sorted_heuristic_feedback,f"Preference_guide_Data/{game_name}/{human_subject_id}/{human_subject_id}_heuristic_feedback.pt")