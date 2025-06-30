import matplotlib.pyplot as plt
import json
import numpy as np


result = {}
for i in range(6):
    result[i] = [0 for _ in range(30)]


num_of_seed = 5
result_list = []

for i in ['42','43','44','45','46']:

    with open(f'/home/grl/Desktop/Explore_data_top_15/seed_{i}_explore_results_g.json', 'r') as f:
        explore = json.load(f)
    single_seed_rsult_list = []
    
    for weight_key, info_dict in explore.items():
        sum_list = [0 for _ in range(30)]
        count_list = [0 for _ in range(30)]
        for episode_key,val_list in info_dict.items():
            for i,val in enumerate(val_list):
                sum_list[i] += val
                count_list[i] += 1

        final_list = [sum_val/count_val/10000 for sum_val,count_val in zip(sum_list,count_list)]        
        
        if len(final_list) < 30:
            for i in range(30-len(final_list)):
                final_list.append(0)
        single_seed_rsult_list.append(final_list)
    result_list.append(single_seed_rsult_list)

result_array1 = np.array(result_list)     
        
result = {}
for i in range(15):
    result[i] = [0 for _ in range(30)]


num_of_seed = 5
result_list = []

for i in ['01', '02', '07',"10","13","15", '17', '18', '25', '30', '32', '33', '37', '39', '44']:

    with open(f'/home/grl/Desktop/Explore_data_top_15/{i}_explore_results_g.json', 'r') as f:
        explore = json.load(f)
    single_seed_rsult_list = []
    
    for weight_key, info_dict in explore.items():
        sum_list = [0 for _ in range(30)]
        count_list = [0 for _ in range(30)]
        for episode_key,val_list in info_dict.items():
            for i,val in enumerate(val_list):
                sum_list[i] += val
                count_list[i] += 1

        final_list = [sum_val/count_val/10000 for sum_val,count_val in zip(sum_list,count_list)]        
        
        if len(final_list) < 30:
            for i in range(30-len(final_list)):
                final_list.append(0)
        single_seed_rsult_list.append(final_list)
    result_list.append(single_seed_rsult_list)

result_array = np.array(result_list)     

result_array = np.transpose(result_array, (1,0,2))
result_array1 = np.transpose(result_array1, (1,0,2))


weight_id = range(6)
orange_color_set = [
    "#FFA500",
    "#FFA500",
    "#FFA500",
    "#FFA500",
    "#FFA500",
    "#FFA500"]

# List of purple color codes from light to dark
purple_color_set = [
    "#800080",
    "#800080",
    "#800080" ,
    "#800080" ,
    "#800080" ,
    "#800080" 
]

lw = 4
for i in weight_id:
    plt.figure(figsize=(3,4.5))
    mean_values = np.mean(result_array1[i,:,:],axis = 0)
    std_values = np.std(result_array1[i,:,:],axis=0)
    
    mean_values1 = np.mean(result_array[i,:,:],axis = 0)
    std_values1 = np.std(result_array[i,:,:],axis=0)


    # mean_values1 = np.mean(result_array[weight_id,:,:],axis = 0)
    # std_values1 = np.std(result_array[weight_id,:,:],axis=0)
    if i == 0:
        plt.plot(mean_values.transpose(), label='DDPG',color=purple_color_set[i],linewidth=lw)
        plt.fill_between(range(len(mean_values)), mean_values - std_values, mean_values + std_values, alpha=0.3,color="purple")
    else:
        plt.plot(mean_values.transpose(),color=purple_color_set[i],linewidth=lw)
        plt.fill_between(range(len(mean_values)), mean_values - std_values, mean_values + std_values, alpha=0.3,color="purple")

# plt.plot(mean_values1, label='GUIDE',color="orange")
# plt.fill_between(range(len(mean_values1)), mean_values1 - std_values1, mean_values1 + std_values1, alpha=0.3,color="orange")
# plt.ylim(0.2,1)
    if i == 0:
        plt.plot(mean_values1.transpose(), label='GUIDE',color=orange_color_set[i],linewidth=lw)
        plt.fill_between(range(len(mean_values1)), mean_values1 - std_values1, mean_values1 + std_values1, alpha=0.3,color="orange")
    else:
        plt.plot(mean_values1.transpose(),color=orange_color_set[i],linewidth=lw)
        plt.fill_between(range(len(mean_values1)), mean_values1 - std_values1, mean_values1 + std_values1, alpha=0.3,color="orange")

    # plt.title(f'{i*2} minutes')    plt.xlabel('Steps')

    
    if i == 0:
        # plt.title(f'Explore Rate at {i*2} minute')
        # plt.ylabel('Visable Area Ratio')
        # plt.tight_layout()
        plt.legend(loc='upper left')
    plt.ylim(0.2,0.85)
    plt.grid()
    plt.savefig(f'/home/grl/Desktop/Explore_data_top_15/{i*2}_explore_rate.png',dpi = 300)