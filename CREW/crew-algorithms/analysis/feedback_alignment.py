import pickle
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests


def get_feedback(sub_path):
    heu_path = f'../Data/FINAL/done_516/{sub_path}/heuristic_values.pkl'
    hf_path = f'../Data/FINAL/done_516/{sub_path}/hf_values.pkl'

    with open(heu_path, 'rb') as f:
        heu = pickle.load(f)
    with open(hf_path, 'rb') as f:
        hf = pickle.load(f)
    return hf, heu

def process(feedback):
    new_feedback = []
    f_max, f_min, f_mean, f_std = max(feedback), min(feedback), np.mean(feedback), np.std(feedback)
    for f in feedback:
        new_feedback.append((f - f_mean)/(f_std))
        # if f > 0:
        #     # new_feedback.append(f/f_max)
        #     new_feedback.append((f- f_mean)/(f_std))
        # else:
        #     new_feedback.append((f - f_mean)/(f_std))
        #     # new_feedback.append(-f/f_min)
    return new_feedback


def affine_invariant_sequence_similarity(hf, heu, shift=2):
    hf = process(hf)
    heu = process(heu)

    # plt.plot(hf)
    # plt.plot(heu)
    # plt.show()
    
    if shift == 0:
        hf = np.array(hf)
        heu = np.array(heu)
    else:
        hf = np.array(hf[shift:])
        heu = np.array(heu[:-shift])

    error = np.sum(np.abs(hf - heu)) / len(hf)
    print(error)

    return error

def granger_causality(hf, heu):
    x = np.array([hf, heu]).T
    return grangercausalitytests(x, [2], addconst=True, verbose=False)[2][0]['ssr_ftest'][1]


def get_plot(sub_path):
    hf, heu = get_feedback(sub_path)
    shifts = []

    for s in range(1):
        shifts.append(affine_invariant_sequence_similarity(hf, heu, s))
        # out = granger_causality(hf, heu)
        # print(out)
        # shifts.append(out)
    return shifts
    # plt.plot(shifts)
    # plt.show()


# path5 = '05_data/saved_training/find_treasure_guide'
# path1 = '32_data/saved_training/0515_1425_find_treasure_guide'
# path2 = '31_data/saved_training/0515_1821_find_treasure_guide'
# path3 = '33_data/saved_training/0515_1426_find_treasure_guide'
# path4 = '36_data/saved_training/0515_1327_find_treasure_guide'

# p1 = get_plot(path1)
# p2 = get_plot(path2)
# p3 = get_plot(path3)
# p4 = get_plot(path4)
# p5 = get_plot(path5)
# # plt.plot(p1, label='32')
# # plt.plot(p5, label='05')
# # plt.plot(p2, label='31')
# # plt.plot(p3, label='33')
# # plt.plot(p4, label='36')
# plt.legend()
# plt.show()