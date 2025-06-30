import os
from time import time
import json

        
def eval():
    ddpg_list = os.listdir('../Data/FINAL/rl_baselines/ddpg/')
    sac_list = os.listdir('../Data/FINAL/rl_baselines/sac/')
    print(ddpg_list)
    print(sac_list)

    for exp in sac_list:
        if 'bowling' in exp:
            print(exp)
            os.system('python crew_algorithms/sac/eval.py envs=bowling envs.time_scale=1 exp_path=rl_baselines/sac/' + exp)
        elif 'find_treasure' in exp:
            print(exp)
            os.system('python crew_algorithms/sac/eval.py envs=find_treasure envs.time_scale=1 exp_path=rl_baselines/sac/' + exp)
        elif 'hide_and_seek_1v1' in exp:
            print(exp)
            os.system('python crew_algorithms/sac/eval.py envs=hide_and_seek_1v1 envs.time_scale=1 exp_path=rl_baselines/sac/' + exp)


    for exp in ddpg_list:
        if 'bowling' in exp:
            print(exp)
            os.system('python crew_algorithms/ddpg/eval.py envs=bowling envs.time_scale=1 exp_path=rl_baselines/ddpg/' + exp)
        elif 'find_treasure' in exp:
            print(exp)
            os.system('python crew_algorithms/ddpg/eval.py envs=find_treasure envs.time_scale=1 exp_path=rl_baselines/ddpg/' + exp)
        elif 'hide_and_seek_1v1' in exp:
            print(exp)
            os.system('python crew_algorithms/ddpg/eval.py envs=hide_and_seek_1v1 envs.time_scale=1 exp_path=rl_baselines/ddpg/' + exp)


def make_scores(seed):
    sac_list = os.listdir('../Data/FINAL/rl_baselines/sac')
    ddpg_list = os.listdir('../Data/FINAL/rl_baselines/ddpg')
    for exp in ddpg_list:
        if exp[-2:] != seed:
            continue
        if 'bowling' in exp:
            with open('../Data/FINAL/rl_baselines/ddpg/' + exp + '/results.json', 'r') as f:
                bowl = json.load(f)
        elif 'find_treasure' in exp and 'heuristic' not in exp:
            with open('../Data/FINAL/rl_baselines/ddpg/' + exp + '/results.json', 'r') as f:
                find = json.load(f)
        elif 'find_treasure' in exp and 'heuristic' in exp:
            with open('../Data/FINAL/rl_baselines/ddpg/' + exp + '/results.json', 'r') as f:
                find_h = json.load(f)
        elif 'hide_and_seek_1v1' in exp and 'heuristic' not in exp:
            with open('../Data/FINAL/rl_baselines/ddpg/' + exp + '/results.json', 'r') as f:
                hide = json.load(f)
        elif 'hide_and_seek_1v1' in exp and 'heuristic' in exp:
            with open('../Data/FINAL/rl_baselines/ddpg/' + exp + '/results.json', 'r') as f:
                hide_h = json.load(f)

    for exp in sac_list:
        if exp[-2:] != seed:
            continue
        if 'bowling' in exp:
            with open('../Data/FINAL/rl_baselines/sac/' + exp + '/results.json', 'r') as f:
                bowl_s = json.load(f)
        elif 'find_treasure' in exp:
            with open('../Data/FINAL/rl_baselines/sac/' + exp + '/results.json', 'r') as f:
                find_s = json.load(f)
        elif 'hide_and_seek_1v1' in exp:
            with open('../Data/FINAL/rl_baselines/sac/' + exp + '/results.json', 'r') as f:
                hide_s = json.load(f)

    print(bowl)
    print(bowl_s)

    print(find)
    print(find_h)
    print(find_s)

    print(hide)
    print(hide_h)
    print(hide_s)


    final_results = {
        'ddpg': {
            'bowling': [bowl[b] for b in bowl.keys()],
            'find_treasure': [find[f] for f in find.keys()],
            'hide_and_seek_1v1': [hide[h] for h in hide.keys()],
        },
        'ddpg_heuristic': {
            'find_treasure': [find_h[f] for f in find_h.keys()],
            'hide_and_seek_1v1': [hide_h[h] for h in hide_h.keys()],          
        },
        'sac': {
            'bowling': [bowl_s[b] for b in bowl_s.keys()],
            'find_treasure': [find_s[f] for f in find_s.keys()],
            'hide_and_seek_1v1': [hide_s[h] for h in hide_s.keys()],
        }
    }

    for method in final_results.keys():
        for env in final_results[method].keys():
            print(method, env, len(final_results[method][env]))
        
    with open('../Data/FINAL/rl_baselines/final_results_'+ seed + '.json', 'w') as f:
        json.dump(final_results, f)

if __name__ == "__main__":
    # eval()
    for seed in ['52', '53', '54', '55', '56']:
        make_scores(seed)