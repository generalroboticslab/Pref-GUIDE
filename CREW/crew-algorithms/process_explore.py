import os
from time import time
import json

def eval(data_path):
    # tamer_list = os.listdir('../Data/FINAL/' + data_path + '/deep_tamer')
    guide_list = os.listdir('../Data/FINAL/' + data_path + '/guide')

    guide_list = [filename for filename in guide_list if 'continue' not in filename]

    print(guide_list)

    # for exp in tamer_list:
    #     # if 'bowling' in exp:
    #     #     print(exp)
    #     #     os.system('python crew_algorithms/deep_tamer/eval.py envs=bowling envs.time_scale=1 exp_path=' + data_path + '/deep_tamer/' + exp)
    #     if 'find_treasure' in exp:
    #         print(exp)
    #         os.system('python crew_algorithms/deep_tamer/eval_explore.py envs=find_treasure envs.time_scale=1 exp_path=' + data_path + '/deep_tamer/' + exp)
    # #     elif 'hide_and_seek_1v1' in exp:
    #         # print(exp)
    #         # os.system('python crew_algorithms/deep_tamer/eval_explore.py envs=hide_and_seek_1v1 envs.time_scale=1 exp_path=' + data_path + '/deep_tamer/' + exp)


    for exp in guide_list:
        # if 'bowling' in exp:
        #     print(exp)
        #     os.system('python crew_algorithms/ddpg/eval.py envs=bowling envs.time_scale=1 exp_path=' + data_path + '/guide/' + exp)
        if 'find_treasure' in exp:

            print(exp)
            os.system('python crew_algorithms/ddpg/eval_explore.py envs=find_treasure envs.time_scale=1 exp_path=' + data_path + '/guide/' + exp)
        # elif 'hide_and_seek_1v1' in exp:
            # print(exp)
            # os.system('python crew_algorithms/ddpg/eval_explore.py envs=hide_and_seek_1v1 envs.time_scale=1 exp_path=' + data_path + '/guide/' + exp)


# def make_scores(data_path):
#     tamer_list = os.listdir('../Data/FINAL/' + data_path + '/deep_tamer')
#     guide_list = os.listdir('../Data/FINAL/' + data_path + '/guide')
#     for exp in guide_list:
#         # if 'bowling' in exp and 'continue' not in exp:
#         #     with open('../Data/FINAL/' + data_path + '/guide/' + exp + '/results.json', 'r') as f:
#         #         bowl = json.load(f)
#         # elif 'bowling' in exp and 'continue' in exp:
#         #     with open('../Data/FINAL/' + data_path + '/guide/' + exp + '/results.json', 'r') as f:
#         #         bowl_c = json.load(f)
#         if 'find_treasure' in exp and 'continue' not in exp:
#             with open('../Data/FINAL/' + data_path + '/guide/' + exp + '/explore_results.json', 'r') as f:
#                 find = json.load(f)
#         elif 'find_treasure' in exp and 'continue' in exp:
#             with open('../Data/FINAL/' + data_path + '/guide/' + exp + '/explore_results.json', 'r') as f:
#                 find_c = json.load(f)
#         elif 'hide_and_seek_1v1' in exp and 'continue' not in exp:
#             with open('../Data/FINAL/' + data_path + '/guide/' + exp + '/explore_results.json', 'r') as f:
#                 hide = json.load(f)
#         elif 'hide_and_seek_1v1' in exp and 'continue' in exp:
#             with open('../Data/FINAL/' + data_path + '/guide/' + exp + '/explore_results.json', 'r') as f:
#                 hide_c = json.load(f)

#     # bowl_0 = bowl['0']
#     find_0 = find['0']
#     hide_0 = hide['0']

#     for exp in tamer_list:
#         # if 'bowling' in exp:
#         #     with open('../Data/FINAL/' + data_path + '/deep_tamer/' + exp + '/results.json', 'r') as f:
#         #         bowl_t = json.load(f)
#         if 'find_treasure' in exp:
#             with open('../Data/FINAL/' + data_path + '/deep_tamer/' + exp + '/explore_results.json', 'r') as f:
#                 find_t = json.load(f)
#         elif 'hide_and_seek_1v1' in exp:
#             with open('../Data/FINAL/' + data_path + '/deep_tamer/' + exp + '/explore_results.json', 'r') as f:
#                 hide_t = json.load(f)

#     # print(bowl)
#     # print(bowl_c)
#     print(find)
#     print(find_c)
#     print(hide)
#     print(hide_c)
#     # print(bowl_t)
#     print(find_t)
#     print(hide_t)


    # final_results = {
    #     'guide': {
    #         # 'bowling': [bowl[b] for b in bowl.keys()] + [bowl_c[b] for b in bowl_c.keys()][1:],
    #         'find_treasure': [find[f] for f in find.keys()] + [find_c[f] for f in find_c.keys()][1:],
    #         'hide_and_seek_1v1': [hide[h] for h in hide.keys()] + [hide_c[h] for h in hide_c.keys()][1:],
    #     },
    #     'deep_tamer': {
    #         # 'bowling': [bowl_0] + [bowl_t[b] for b in bowl_t.keys()][1:],
    #         'find_treasure': [find_0] + [find_t[f] for f in find_t.keys()][1:],
    #         'hide_and_seek_1v1': [hide_0] + [hide_t[h] for h in hide_t.keys()][1:],
    #     }
    # }

    # for method in final_results.keys():
    #     for env in final_results[method].keys():
    #         print(method, env, len(final_results[method][env]))
        
    # with open('../Data/FINAL/' + data_path + '/final_results.json', 'w') as f:
    #     json.dump(final_results, f)


def process_data(data_path):
    tic = time()
    eval(data_path)
    # make_scores(data_path)
    print('Time:', time() - tic)

# def check(data_path):
#     with open('../Data/FINAL/' + data_path + '/final_results.json', 'r') as f:
#         results = json.load(f)
#     if len(results['guide']['hide_and_seek_1v1']) == 16 and len(results['deep_tamer']['hide_and_seek_1v1']) == 6:
#         print(data_path + ' All good')
#     else:
#         print(data_path + 'Missing' + str(len(results['guide']['hide_and_seek_1v1'])) + str(len(results['deep_tamer']['hide_and_seek_1v1'])))

if __name__ == "__main__":
    # indiceies = [f"{i:02}" for i in range(1, 2)]
    # indiceies = ['02', '03', '04', '07']
    # indiceies = [14, 19, 23, 26, 30]
    # indiceies = ['47']#
    # indiceies = ['48', '50', '51']
    # indiceies = [str(i) for i in indiceies]
    # indiceies = ['11'], 
    # indiceies = ['11', '12', '13', '14', '15', '17', '18', '19', '20', '21', '22', '23', '24', '25']
    # indiceies = ['41', '43', '44', '45', '46', '47', '48', '50', '51']
    indiceies = ['96']
    for i in indiceies:
        process_data(i + '_data')
        # check(i + '_data')