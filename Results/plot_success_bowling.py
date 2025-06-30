import os 
import matplotlib.pyplot as plt
import numpy as np
import json


game_name = "bowling"

testing_seed = 47

training_seed = [37, 38, 39, 40, 41, 42, 43, 44, 45, 46]

subject_list = list(range(52))
subject_list.remove(42)
subject_list.remove(12)

# subject_list = [1,2,7,10,13,15,17,18,25,30,32,33,37,39,44]

eval_weights = 20

data_dict = {
    # "baseline": "baseline",
    "GUIDE (human-in-the-loop)": "../GUIDE_TRAINING_RESULTS" ,
    "GUIDE": "guide_model_0.0001",

    "Pref-GUIDE": "hard_label_pretrain_activation_0.0001",
    "Pref-GUIDE Voting": "vote_ref_pretrain_activation_0.0001",
    "DDPG": "baseline_0.0001" , 
    # "DDPG Heuristic": "heuristic_0.0001",
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
}

result_list = []

for key, value in data_dict.items():
    
    if key.find("DDPG") != -1:
        data_path = os.path.join("ddpg_cont",game_name,value)

        result_array = np.zeros((len(training_seed), eval_weights+1))
        for i, seed in enumerate(training_seed):
            folder_path = os.path.join(data_path, f"seed_{seed}/results-47.json")
            data = json.load(open(folder_path,"r"))
            
            for j in range(eval_weights):
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

            result_array = np.zeros((len(subject_list), eval_weights+1-5))
            
            
            for i, subject in enumerate(subject_list):
                folder_path = os.path.join(data_path, f"subject_{subject}")
                if not os.path.exists(folder_path):
                    print(f"Folder {folder_path} does not exist.")
                    continue
                # Load the data
                data = json.load(open(os.path.join(folder_path, f"results-{testing_seed}.json"), "r"))
                # Extract the data
                for j in range(eval_weights-5):
                    try:
                        result_array[i,j+1] = data[f"{j+1}"]
                    except KeyError:
                        print(f"Key {j} not found in data for subject {subject} in {key}")
                        result_array[i,j] = 0
                # breakpoint()
                result_array[i,0] = result_list[0][i,5] / 10
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
    print(result_list[i].shape)
    if result_list[i].shape[-1] == eval_weights + 1:
        mean_list[i][0] = mean_list[0][0]
        std_list[i][0] = std_list[0][0]
    




plt.figure(figsize=(20,20))
ax = plt.gca()  
# Setting Background colour yellow 
ax.set_facecolor("#EAEAF2")
plt.xlim(0,eval_weights)
plt.ylim(5,70)

length = eval_weights + 1

plt.yticks(fontsize=48)
plt.xticks(range(0,length,5),fontsize=48)
# plt.title(f"{game_name}",fontsize=24)


for i in range(len(mean_list)):
    if len(mean_list[i]) == 31:
        mean_list[i][0] = mean_list[0][0]
        std_list[i][0] == std_list[0][0]
        print(mean_list[0][0])
        
key_list = list(data_dict.keys())

for i in range(len(mean_list)):
    if len(mean_list[i]) == eval_weights + 1:
        plt.plot(range(0,length),mean_list[i], marker='o', linewidth=12, color=color_dict[key_list[i]])
    elif len(mean_list[i]) == 6:
        plt.plot(range(0,6),mean_list[i], marker='o', linewidth=12, color=color_dict[key_list[i]],linestyle="--")
    else:
        plt.plot(range(5,eval_weights + 1),mean_list[i], marker='o', linewidth=12, color=color_dict[key_list[i]])
# plt.legend(data_dict.keys(), loc='lower right',fontsize=36)

for i in range(len(mean_list)):
    lower = mean_list[i] - std_list[i]
    upper = mean_list[i] + std_list[i]
    if len(mean_list[i]) == eval_weights + 1:
        plt.fill_between(range(0,length), lower, upper, alpha=0.2, color=color_dict[key_list[i]])
    elif len(mean_list[i]) == 6:
        plt.fill_between(range(0,6), lower, upper, alpha=0.2, color=color_dict[key_list[i]])
    else:
        plt.fill_between(range(5,eval_weights + 1), lower, upper, alpha=0.2, color=color_dict[key_list[i]])

plt.tick_params(width=3)  # Thicker ticks
plt.gca().spines[:].set_linewidth(3)  # Apply to all spines (axis lines)
plt.axvline(x=5, color='black', linestyle='--', linewidth=7)

ax.grid(True, color='white', linewidth=4)
# plt.xlabel("Minute", fontsize=36)
# plt.ylabel("Average Score", fontsize=36)
# plt.xticks(fontsize=24)
plt.savefig(f"figure/{game_name}_{len(subject_list)}.png")

    
        
        

    