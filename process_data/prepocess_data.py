from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
# from utils import plot_feedback, save_image1
import torch
import os

human_subject_id_list = list(range(52))
human_subject_id_list.remove(42)
human_subject_id_list.remove(12)

S = 20000

game_name = "hide_and_seek_1v1"

for human_subject_id in human_subject_id_list:
    human_subject_id = str(human_subject_id) if human_subject_id >= 10 else "0" + str(human_subject_id)
    print(human_subject_id)
    storage = LazyMemmapStorage(S)

    rb = TensorDictReplayBuffer(storage = storage)
    try:
        rb.loads(f"RL_checkpoint/{human_subject_id}_Data/saved_training/{human_subject_id}_{game_name}_guide/prb.pkl")
    except:
        rb.loads(f"RL_checkpoint/{human_subject_id}_data/saved_training/{human_subject_id}_{game_name}_guide/prb.pkl")
    data = rb.sample(S)

    index = data["index"]
    traj_id = data["collector","traj_ids"]
    sorted_index = index[index.argsort()].unique()

    data_length = len(sorted_index)
    sorted_traj_id = torch.zeros(data_length)

    index_list = []
    new_index_list = []
    for i in range(len(index.argsort())):
        if index[index.argsort()[i]].item() in new_index_list:
            continue
        new_index_list.append(index[index.argsort()[i]].item())
        index_list.append(index.argsort()[i].item())

    for i,ind in enumerate(index_list):
        # print(i,ind)
        sorted_traj_id[i] = traj_id[ind]
    print(sorted_traj_id.shape)

    if not os.path.exists(f"Preference_guide_Data/{game_name}/{human_subject_id}"):
        os.makedirs(f"Preference_guide_Data/{game_name}/{human_subject_id}")

    torch.save(sorted_traj_id,f"Preference_guide_Data/{game_name}/{human_subject_id}/{human_subject_id}_tarj_ids.pt")