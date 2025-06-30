import matplotlib.pyplot as plt
import json


result1 = {}
for i in range(3):
    result1[i] = [0 for _ in range(30)]


for i in ['99']:

    with open(f'/home/grl/Desktop/Explore_data_top_15/{i}_explore_results_g.json', 'r') as f:
        explore = json.load(f)

    sum_list = [0 for _ in range(30)]
    count_list = [0 for _ in range(30)]

    for episode_key,val_list in explore['2'].items():
        if int(episode_key) >0 and int(episode_key) < 11:
            for i,val in enumerate(val_list):
                sum_list[i] += val
                count_list[i] += 1
    
    final_list99 = [sum_val/count_val/10000 for sum_val,count_val in zip(sum_list,count_list)]
            # for i in range(30):
            #     result1[int(weight_key)][i] += final_list[i] 
print(final_list99)

result = {}
for i in range(3):
    result[i] = [0 for _ in range(30)]


for i in ['37', '18', '15', '17', '32', '19', '04', '27', '02', '01', '10', '30', '07', '39', '44', '25']:

    with open(f'/home/grl/Desktop/Explore_data_top_15/{i}_explore_results_dt.json', 'r') as f:
        explore = json.load(f)

    for weight_key, info_dict in explore.items():
        if int(weight_key) < 3:
            sum_list = [0 for _ in range(30)]
            count_list = [0 for _ in range(30)]
            for episode_key,val_list in info_dict.items():
                for i,val in enumerate(val_list):
                    sum_list[i] += val
                    count_list[i] += 1

            final_list = [sum_val/count_val/10000 for sum_val,count_val in zip(sum_list,count_list)]
            for i in range(30):
                result[int(weight_key)][i] += final_list[i] 




plt.figure(figsize=(10, 6))
plt.stackplot(range(30),final_list99, labels=['DDPG Weight 1'])

# Plot the top plot (result[1]) on top of the bottom one
plt.stackplot(range(30), [a/16 for a in result[2]], labels=['GUIDE Weight 1'])
plt.title('Stacked Plot for result1')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()