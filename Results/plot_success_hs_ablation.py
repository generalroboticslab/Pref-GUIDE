import os 
import matplotlib.pyplot as plt
import numpy as np
import json

# game_name = "find_treasure"
# game_name = "bowling"
game_name = "hide_and_seek_1v1"
testing_seed = 47

training_seed = list(range(42,47))

# subject_list = list(range(52))
# subject_list.remove(42)
# subject_list.remove(12)

subject_list = [1,2,7,10,15,17,18,25,30,32,33,37,39,44]

eval_weights = 25

data_dict = {
    # "baseline": "baseline",
    "GUIDE (human-in-the-loop)": "../GUIDE_TRAINING_RESULTS" ,
    "GUIDE": "guide_model_0.0001",

    "Pref-GUIDE": "hard_label_pretrain_activation_0.0001",
    "Pref-GUIDE\nw/o No_preference Range": "hard_label_pretrain_activation_no_random_window_0.0001",
    "Pref-GUIDE\nw/o Moving Window Sampling": "hard_label_pretrain_activation_no_moving_window_0.0001",
    "Pref-GUIDE\nw/o No_preference Range\nw/o Moving Window Sampling": "hard_label_pretrain_activation_no_random_window_no_moving_window_0.0001",
    
    # "Pref-GUIDE Voting": "vote_ref_pretrain_activation_0.0001",
    # "Pref-GUIDE Voting\nw/o Normalization": "vote_ref_pretrain_activation_majority_vote_0.0001",
}

color_dict = {
    "DDPG": "C0" , 
    "GUIDE (human-in-the-loop)": "C1" ,
    "GUIDE": "C1",
    "Pref-GUIDE": "C2",
    "Pref-GUIDE Voting": "C3", 
    "DDPG Heuristic": "C4",
    "Pref-GUIDE Hard Label": "C3",
    "Pref-GUIDE Soft Label": "C2",
    "Pref-GUIDE\nw/o No_preference Range": "C5",
    "Pref-GUIDE\nw/o Moving Window Sampling": "C6",
    "Pref-GUIDE\nw/o No_preference Range\nw/o Moving Window Sampling": "C9",
    "Pref-GUIDE Voting\nw/o Normalization": "grey"
}

result_list = []

for key, value in data_dict.items():
    
    if key.find("DDPG") != -1:
        data_path = os.path.join("ddpg_cont",game_name,value)

        result_array = np.zeros((len(training_seed), 31))
        for i, seed in enumerate(training_seed):
            folder_path = os.path.join(data_path, f"seed_{seed}/results-47.json")
            data = json.load(open(folder_path,"r"))
            
            for j in range(30):
                result_array[i,j+1] = data[f"{j+1}"]
        result_list.append(result_array)
    else:
    
        if key.find("human") != -1:
            data_path = os.path.join("ddpg_cont",game_name,value)

            result_array = np.zeros((len(subject_list), 6))
            for i, subject in enumerate(subject_list):
                subject = f"0{subject}" if subject < 10 else subject  
                folder_path = os.path.join(data_path, f"{subject}_final_results.json")
                
                data = json.load(open(folder_path,"r"))
                data = data["guide"][game_name]
                
                for j in range(6):
                    try:
                        result_array[i,j] = data[j]
                    except KeyError:
                        print(f"Key {j} not found in data for subject {subject} in {key}")
                        result_array[i,j] = 0
            result_list.append(result_array)
            
        else:
            data_path = os.path.join("ddpg_cont",game_name,value)

            result_array = np.zeros((len(subject_list), eval_weights+1))
            
            
            for i, subject in enumerate(subject_list):
                folder_path = os.path.join(data_path, f"subject_{subject}")
                if not os.path.exists(folder_path):
                    print(f"Folder {folder_path} does not exist.")
                    continue
                # Load the data
                data = json.load(open(os.path.join(folder_path, f"results-{testing_seed}.json"), "r"))
                # Extract the data
                for j in range(eval_weights):
                    try:
                        result_array[i,j+1] = data[f"{j+1}"]
                    except KeyError:
                        print(f"Key {j} not found in data for subject {subject} in {key}")
                        result_array[i,j] = 0
                # breakpoint()
                result_array[i,0] = result_list[0][i,5]
            result_list.append(result_array)
# print(result_list)

if game_name == "bowling":
    for i in range(len(result_list)):
        if i != 0:
            result_list[i] = result_list[i] * 10


mean_list = []
std_list = []
for i in range(len(result_list)):
    mean_list.append(np.mean(result_list[i], axis=0))
    std_list.append(np.std(result_list[i], axis=0))




plt.figure(figsize=(20,20))
ax = plt.gca()  
# Setting Background colour yellow 
ax.set_facecolor("#EAEAF2")
plt.xlim(0,20)

if game_name == "find_treasure":
    plt.ylim(10,100)
elif game_name == "hide_and_seek_1v1":
    plt.ylim(0,100)
elif game_name == "bowling":
    plt.ylim(-5,85)

length = 10 + 2*eval_weights + 1

plt.yticks(fontsize=48)
plt.xticks(range(0,length,10),fontsize=48)
# plt.title(f"{game_name}",fontsize=24)


for i in range(len(mean_list)):
    if len(mean_list[i]) == 31:
        mean_list[i][0] = mean_list[0][0]
        std_list[i][0] == std_list[0][0]
        print(mean_list[0][0])
        
key_list = list(data_dict.keys())

for i in range(len(mean_list)):
    if len(mean_list[i]) == 26:
        plt.plot(range(10,length,2),mean_list[i], marker='o', linewidth=12, color=color_dict[key_list[i]])
    elif len(mean_list[i]) == 6:
        plt.plot(range(0,11,2),mean_list[i], marker='o', linewidth=12, color=color_dict[key_list[i]],linestyle="--")
    else:
        plt.plot(range(0,61,2),mean_list[i], marker='o', linewidth=12, color=color_dict[key_list[i]])
plt.legend(data_dict.keys(), loc='lower right',fontsize=40)

for i in range(len(mean_list)):
    lower = mean_list[i] - std_list[i]
    upper = mean_list[i] + std_list[i]
    if len(mean_list[i]) == 26:
        plt.fill_between(range(10,length,2), lower, upper, alpha=0.2, color=color_dict[key_list[i]])
    elif len(mean_list[i]) == 6:
        plt.fill_between(range(0,11,2), lower, upper, alpha=0.2, color=color_dict[key_list[i]])
    else:
        plt.fill_between(range(0,61,2), lower, upper, alpha=0.2, color=color_dict[key_list[i]])

plt.tick_params(width=3)  # Thicker ticks
plt.gca().spines[:].set_linewidth(3)  # Apply to all spines (axis lines)

if game_name == "find_treasure":
    plt.axvline(x=10, color='black', linestyle='--', linewidth=7)
elif game_name == "hide_and_seek_1v1":
    plt.axvline(x=10, color='black', linestyle='--', linewidth=7)
elif game_name == "bowling":
    plt.axvline(x=5, color='black', linestyle='--', linewidth=7)

ax.grid(True, color='white', linewidth=4)
plt.savefig(f"figure/{game_name}_Ablation1.png")

    
        
        

    