import matplotlib.pyplot as plt
import json
import numpy as np
import os

plot_exp = 'hide_and_seek_1v1'
# plot_exp = 'find_treasure'
# plot_exp = 'bowling'

data = []

file_list = os.listdir('../CREW_RESULTS_ALL/')

for i, file in enumerate(file_list):
    if i not in [8,  9, 13, 16, 17, 19, 22, 28, 31, 45]:
        continue
    # if i not in [16, 17, 19, 22, 28]: # best 5
    #     continue
    with open('../CREW_RESULTS_ALL/' + file, 'r') as f:
        d = json.load(f)
        data.append(d)


num_data = len(data)
deep_tamer_bowl = []
guide_bowl = []
deep_tamer_find = []
guide_find = []
deep_tamer_hide = []
guide_hide = []

for i,d in enumerate(data):
    deep_tamer_bowl.append(d['deep_tamer']['bowling'])
    guide_bowl.append(d['guide']['bowling'])
    deep_tamer_find.append(d['deep_tamer']['find_treasure'])
    guide_find.append(d['guide']['find_treasure'])
    deep_tamer_hide.append(d['deep_tamer']['hide_and_seek_1v1'])
    guide_hide.append(d['guide']['hide_and_seek_1v1'])

ddpg_bowl = []
ddpg_find = []
ddpg_hide = []
sac_bowl = []
sac_find = []
sac_hide = []
heuristic_find = []
heuristic_hide = []

for seed in [str(i) for i in range(47, 62)]:
    with open('../RL_RESULTS/final_results_' + seed + '.json', 'r') as f:
        data = json.load(f)
        ddpg_bowl.append(data['ddpg']['bowling'])
        ddpg_find.append(data['ddpg']['find_treasure'])
        ddpg_hide.append(data['ddpg']['hide_and_seek_1v1'])
        sac_bowl.append(data['sac']['bowling'])
        sac_find.append(data['sac']['find_treasure'])
        sac_hide.append(data['sac']['hide_and_seek_1v1'])
        heuristic_find.append(data['ddpg_heuristic']['find_treasure'])
        heuristic_hide.append(data['ddpg_heuristic']['hide_and_seek_1v1'])


ddpg_bowl = np.array(ddpg_bowl) * 10
sac_bowl = np.array(sac_bowl)
ddpg_find = np.array(ddpg_find)
sac_find = np.array(sac_find)
heuristic_find = np.array(heuristic_find)
ddpg_hide = np.array(ddpg_hide)
sac_hide = np.array(sac_hide)
heuristic_hide = np.array(heuristic_hide)


ddpg_bowl_means = np.mean(ddpg_bowl, axis=0)
ddpg_bowl_std = np.std(ddpg_bowl, axis=0)
sac_bowl_means = np.mean(sac_bowl, axis=0)
sac_bowl_std = np.std(sac_bowl, axis=0)
ddpg_find_means = np.mean(ddpg_find, axis=0)
ddpg_find_std = np.std(ddpg_find, axis=0)
sac_find_means = np.mean(sac_find, axis=0)
sac_find_std = np.std(sac_find, axis=0)
heuristic_find_means = np.mean(heuristic_find, axis=0)
heuristic_find_std = np.std(heuristic_find, axis=0)
ddpg_hide_means = np.mean(ddpg_hide, axis=0)
ddpg_hide_std = np.std(ddpg_hide, axis=0)
sac_hide_means = np.mean(sac_hide, axis=0)
sac_hide_std = np.std(sac_hide, axis=0)
heuristic_hide_means = np.mean(heuristic_hide, axis=0)
heuristic_hide_std = np.std(heuristic_hide, axis=0)



max_scores = [
    # np.array(deep_tamer_bowl).max(axis=1),
    np.array(guide_bowl).max(axis=1),
    # np.array(deep_tamer_find).max(axis=1),
    np.array(guide_find).max(axis=1),
    # np.array(deep_tamer_hide).max(axis=1),
    np.array(guide_hide).max(axis=1),
]
max_scores = np.array(max_scores).sum(axis=0)

print(max_scores)


deep_tamer_bowl = np.array(deep_tamer_bowl) * 10
guide_bowl = np.array(guide_bowl) * 10
deep_tamer_find = np.array(deep_tamer_find)
guide_find = np.array(guide_find)
deep_tamer_hide = np.array(deep_tamer_hide)
guide_hide = np.array(guide_hide)

deep_tamer_bowl_means = np.mean(deep_tamer_bowl, axis=0)
deep_tamer_bowl_std = np.std(deep_tamer_bowl, axis=0)
guide_bowl_means = np.mean(guide_bowl, axis=0)
guide_bowl_std = np.std(guide_bowl, axis=0)
deep_tamer_find_means = np.mean(deep_tamer_find, axis=0)
deep_tamer_find_std = np.std(deep_tamer_find, axis=0)
guide_find_means = np.mean(guide_find, axis=0)
guide_find_std = np.std(guide_find, axis=0)
deep_tamer_hide_means = np.mean(deep_tamer_hide, axis=0)
deep_tamer_hide_std = np.std(deep_tamer_hide, axis=0)
guide_hide_means = np.mean(guide_hide, axis=0)
guide_hide_std = np.std(guide_hide, axis=0)

# print(deep_tamer_bowl_means.max())
# print(guide_bowl_means.max())
# print(deep_tamer_find_means.max())
# print(guide_find_means.max())
# print(deep_tamer_hide_means.max())
# print(guide_hide_means.max())


# Assuming x-values are the same for all points, for example:
x5 = np.arange(0, 6, 1)
x15 = np.arange(0, 16, 1)


x10 = np.arange(0, 12, 2)
x30 = np.arange(0, 32, 2)


if plot_exp == 'bowling':
    
    div_x = [5 for _ in range(100)]
    div_y = [i for i in range(100)]


    plt.plot(x5, deep_tamer_bowl_means, label='DeepTamer', marker='.', color='cornflowerblue')
    plt.fill_between(x5, deep_tamer_bowl_means - deep_tamer_bowl_std, deep_tamer_bowl_means + deep_tamer_bowl_std, alpha=0.2, color='cornflowerblue')
    plt.plot(x15, guide_bowl_means, label='GUIDE', marker='.', color='orange')
    plt.fill_between(x15, guide_bowl_means - guide_bowl_std, guide_bowl_means + guide_bowl_std, alpha=0.2, color='orange')
    plt.plot(x15, sac_bowl_means, label='SAC', marker='.', color='green')
    plt.fill_between(x15, sac_bowl_means - sac_bowl_std, sac_bowl_means + sac_bowl_std, alpha=0.2, color='green')
    plt.plot(x15, ddpg_bowl_means, label='DDPG', marker='.', color='purple')
    plt.fill_between(x15, ddpg_bowl_means - ddpg_bowl_std, ddpg_bowl_means + ddpg_bowl_std, alpha=0.2, color='purple')


elif plot_exp == 'find_treasure':

    div_x = [10 for _ in range(100)]
    div_y = [i for i in range(100)]

    plt.plot(x10, deep_tamer_find_means, label='DeepTamer', marker='.', color='cornflowerblue')
    plt.fill_between(x10, deep_tamer_find_means - deep_tamer_find_std, deep_tamer_find_means + deep_tamer_find_std, alpha=0.2, color='cornflowerblue')
    plt.plot(x30, guide_find_means, label='GUIDE', marker='.', color='orange')
    plt.fill_between(x30, guide_find_means - guide_find_std, guide_find_means + guide_find_std, alpha=0.2, color='orange')
    plt.plot(x30, sac_find_means, label='SAC', marker='.', color='green')
    plt.fill_between(x30, sac_find_means - sac_find_std, sac_find_means + sac_find_std, alpha=0.2, color='green')
    plt.plot(x30, ddpg_find_means, label='DDPG', marker='.', color='purple')
    plt.fill_between(x30, ddpg_find_means - ddpg_find_std, ddpg_find_means + ddpg_find_std, alpha=0.2, color='purple')
    plt.plot(x30, heuristic_find_means, label='Heuristic', marker='.', color='red')
    plt.fill_between(x30, heuristic_find_means - heuristic_find_std, heuristic_find_means + heuristic_find_std, alpha=0.2, color='red')


elif plot_exp == 'hide_and_seek_1v1':
    div_x = [10 for _ in range(100)]
    div_y = [i for i in range(100)]

    plt.plot(x10, deep_tamer_hide_means, label='DeepTamer', marker='.', color='cornflowerblue')
    plt.fill_between(x10, deep_tamer_hide_means - deep_tamer_hide_std, deep_tamer_hide_means + deep_tamer_hide_std, alpha=0.2, color='cornflowerblue')
    plt.plot(x30, guide_hide_means, label='GUIDE', marker='.', color='orange')
    plt.fill_between(x30, guide_hide_means - guide_hide_std, guide_hide_means + guide_hide_std, alpha=0.2, color='orange')
    plt.plot(x30, sac_hide_means, label='SAC', marker='.', color='green')
    plt.fill_between(x30, sac_hide_means - sac_hide_std, sac_hide_means + sac_hide_std, alpha=0.2, color='green')
    plt.plot(x30, ddpg_hide_means, label='DDPG', marker='.', color='purple')
    plt.fill_between(x30, ddpg_hide_means - ddpg_hide_std, ddpg_hide_means + ddpg_hide_std, alpha=0.2, color='purple')
    plt.plot(x30, heuristic_hide_means, label='Heuristic', marker='.', color='red')
    plt.fill_between(x30, heuristic_hide_means - heuristic_hide_std, heuristic_hide_means + heuristic_hide_std, alpha=0.2, color='red')


plt.plot(div_x, div_y, linestyle='--', color='black')
plt.xlabel('Time (min)')

if plot_exp == 'bowling':
    plt.ylabel('Score')
    plt.title('Bowling (%d subjects)' % num_data)

elif plot_exp == 'find_treasure':
    plt.ylabel('Success Rate (%)')
    plt.title('Find Treasure (%d subjects)' % num_data)
    plt.ylim(15, 70)

elif plot_exp == 'hide_and_seek_1v1':
    plt.ylabel('Success Rate (%)')
    plt.title('Hide and Seek (%d subjects)' % num_data)
    plt.ylim(0, 95)


plt.legend()
plt.grid()
plt.show()