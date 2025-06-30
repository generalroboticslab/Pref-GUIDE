import json
from collections import defaultdict
from time import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from crew_algorithms.ddpg.utils import get_time
import os
import numpy as np


class custom_logger():
    def __init__(self, path, start_time=None, read_mode=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.start_time = start_time

        if read_mode:
            with open(path, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = defaultdict(list)

    def log(self, x_axis, y_axis, x_value, y_value, log_time):
        x_value, y_value = round(float(x_value),4), round(float(y_value),4)
        self.data[x_axis + '-' + y_axis].append(x_value)
        self.data[y_axis + '-' + x_axis].append(y_value)
        if log_time:
            self.data['time-' + y_axis].append(round(time() - self.start_time, 3)) # x_axis
            self.data[y_axis + '-time'].append(y_value)

    def get_curve(self, x_axis, y_axis, smooth=1):
        x, y = self.data[x_axis + '-' + y_axis], self.data[y_axis + '-' + x_axis]
        if smooth > 1:
            x = [np.array(x[max(0, i-smooth): i]).mean() for i in range(len(x))]
            y = [np.array(y[max(0, i-smooth): i]).mean() for i in range(len(y))]
        return x, y

    def save_log(self):
        with open(self.path, 'w') as f:
            json.dump(self.data, f)

def make_plots(log_files, log_names, save_img_path=None):
    if save_img_path is None:
        save_img_path = 'crew_algorithms/ddpg/plots/' + get_time() + '/'
    else:
        save_img_path = 'crew_algorithms/ddpg/plots/' + save_img_path + '/'
    os.makedirs(save_img_path, exist_ok=True)
    loggers = [custom_logger(log_file, read_mode=True) for log_file in log_files]
    for key in [
        'avg_episode_reward',
        'avg_episode_reward_hf',
        'success_rate',
        'feedback_model_loss',
        'feedback_model_loss_val',
        'total_loss',
        "actor_loss",
        "q_loss",
        "alpha_loss",
        "alpha",
        "entropy",
    ]:
        if key in [
            'avg_episode_reward',
            'avg_episode_reward_hf',
            'success_rate',
            'feedback_model_loss',
            'feedback_model_loss_val',]:
            plt.figure(figsize=(15, 10)) 
            for i, logger in enumerate(loggers):
            
                if (key + '-time')  not in logger.data:
                    continue
                if 'feedback_model_loss' in key:
                    x, y = logger.get_curve('time', key, smooth=100)
                else:
                    x, y = logger.get_curve('time', key)
                
                # x = [x_i//60 for x_i in x]
                # print(len(x))
                # if "learned_feedback_deployed-time" in logger.data and ('success_rate' in key):
                #     time, deployed = logger.get_curve('time', 'learned_feedback_deployed')

                #     seps = [i for i, x in enumerate(deployed) if x != deployed[i-1] and i > 0]
                #     time_seps = [0] + [time[i] for i in seps]

   
                #     new_x, new_y = [], []
                #     to_subtract = 0
                #     for m in range(0, len(time_seps), 2):
                #         start, end = time_seps[m], time_seps[min(m+1, len(time_seps)-1)]
                #         for j, t in enumerate(x):
                #             if start <= t <= end:
                #                 new_x.append(t-to_subtract)
                #                 new_y.append(y[j])
                #         to_subtract += end - start
                #         print(start)
                #         # print(to_subtract)
                #     x, y = new_x, new_y


                # print(len(x))

                plt.plot(x, y, label=log_names[i], alpha=0.6)
            if 'success_rate' in key:

                plt.ylim(0, 1)

                
            elif 'avg_episode_reward' in key:
                plt.ylim(-30, 0)
            if 'feedback_model_loss' in key or 'success_rate' in key:
                time, deployed = logger.get_curve('time', 'learned_feedback_deployed')
                # time = [x_i//60 for x_i in time]
                plt.fill_between(time, np.where(np.array(deployed) > 0, 1, 0), color='green', alpha=.05, step='post')
                plt.fill_between(time, np.where(np.array(deployed) <= 0, 1, 0), color='red', alpha=.05, step='post')

            ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x//60))
            ax = plt.gca()
            ax.xaxis.set_major_formatter(ticks_x)
            plt.xlim(0, 15000)
            plt.xlabel('Feedback time (min)')
            plt.ylabel(key)
            plt.legend()
            plt.grid(axis='y', linestyle='--')
            plt.savefig(save_img_path + key + '_time.png')
            plt.clf()


        plt.figure(figsize=(15, 10)) 
        for i, logger in enumerate(loggers):
            if (key + '-steps')  not in logger.data:
                continue
            if 'feedback_model_loss' in key:
                x, y = logger.get_curve('steps', key, smooth=100)
            else:
                x, y = logger.get_curve('steps', key)
            # x = [x_i//1000 for x_i in x]
            plt.plot(x, y, label=log_names[i], alpha=0.6)
        if 'success_rate' in key:
            plt.ylim(0, 1)
        elif 'avg_episode_reward' in key:
            plt.ylim(-30, 0)
        if 'feedback_model_loss' in key or 'success_rate' in key:
            steps, deployed = logger.get_curve('steps', 'learned_feedback_deployed')
            plt.fill_between(steps, np.where(np.array(deployed) > 0, 1, 0), color='green', alpha=.05, step='post')
            plt.fill_between(steps, np.where(np.array(deployed) <= 0, 1, 0), color='red', alpha=.05, step='post')
        
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x//1000))
        ax = plt.gca()
        ax.xaxis.set_major_formatter(ticks_x)
        plt.xlim(0, 20000)
        plt.xlabel('steps (k)')
        plt.ylabel(key)
        plt.legend()
        plt.grid(axis='y', linestyle='--')
        plt.savefig(save_img_path + key + '_steps.png')
        plt.clf()
        
def compute_human_time(total_time, minutes_without_feedback=20):
    period = 10 + minutes_without_feedback
    return (total_time // period) * 10 + min(total_time % period, 10)


def make_bar_plot(means, stds, log_names, colors, save_img_path=None, groups=[0.6, 0.8, 0.9], mode='sr'):
    if save_img_path is None:
        save_img_path = 'crew_algorithms/ddpg/plots/' + get_time() + '/'
    else:
        save_img_path = 'crew_algorithms/ddpg/plots/' + save_img_path + '/'
    os.makedirs(save_img_path, exist_ok=True)


    x = np.arange(len(groups))  # the label locations
    width = 0.15  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained', figsize=(10, 6))

    for i, (mean, std) in enumerate(zip(means, stds)):
        offset = width * multiplier
        if log_names[i] not in ['heuristic', 'base']:
            hf = [compute_human_time(time) for time in mean]
        elif log_names[i] == 'heuristic':
            hf = mean
        else:
            hf = [0 for time in mean]
        

        rects_human = ax.bar(x + offset, hf, width, alpha=1, color=colors[i], edgecolor='black')
        rects = ax.bar(x + offset, [m - f for (m, f) in zip(mean, hf)], width, label=log_names[i], yerr=std, alpha=1, bottom=hf, color=colors[i], capsize=5)
        ax.bar_label(rects, padding=5)
        
        multiplier += 1

    rects_human = ax.bar(0, 0, width, alpha=1, color='white', edgecolor='black', label='feedback')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    
    if mode == 'sr':
        ax.set_ylabel('Time (min)')
        ax.set_xlabel('Success Rate')
        # ax.set_title('Success Rate')
        ax.set_xticks(x + width, groups)
        ax.legend(loc='upper left')
        plt.savefig(save_img_path + 'bar_sr.png')
    else:
        ax.set_ylabel('Success Rate')
        ax.set_xlabel('Time (min)')
        # ax.set_title('Success Rate')
        ax.set_xticks(x + width, groups)
        ax.legend(loc='upper left')
        plt.savefig(save_img_path + 'bar_time.png')




def parse_logs_for_sr(log_files_multiseeds, log_name, groups = [0.4, 0.6, 0.8]):

    loggers = [custom_logger(log_file, read_mode=True) for log_file in log_files_multiseeds]

    seed_srs = [
        {groups[0]: 0, groups[1]: 0},
        {groups[0]: 0, groups[1]: 0},
        {groups[0]: 0, groups[1]: 0},
    ]
    for i, log in enumerate(loggers):
        x, y = log.get_curve('time', 'success_rate')
        group_pointer = 0
        for j, yj in enumerate(y):
            if yj >= groups[group_pointer]:
                seed_srs[i][groups[group_pointer]] = x[j]//60
                group_pointer += 1
                if group_pointer >= len(groups):
                    break
    mean = [round(sum([seed_srs[i][group] for i in range(3)])/3, 1) for group in groups]
    std = [np.std([seed_srs[i][group] for i in range(3)]) for group in groups]
    return {'mean': mean, 'std': std, 'log_name': log_name}


def parse_logs_for_time(log_files_multiseeds, log_name, groups = [30, 60, 90]):

    loggers = [custom_logger(log_file, read_mode=True) for log_file in log_files_multiseeds]

    seed_srs = [
        {groups[0]: 0, groups[1]: 0, groups[2]: 0},
        {groups[0]: 0, groups[1]: 0, groups[2]: 0},
        {groups[0]: 0, groups[1]: 0, groups[2]: 0},
    ]
    for i, log in enumerate(loggers):
        x, y = log.get_curve('time', 'success_rate')
        group_pointer = 0
        for j, xj in enumerate(x):
            if xj/60 >= groups[group_pointer]:
                seed_srs[i][groups[group_pointer]] = y[j]
                group_pointer += 1
                if group_pointer >= len(groups):
                    break
    mean = [round(sum([seed_srs[i][group] for i in range(3)])/3, 2) for group in groups]
    std = [np.std([seed_srs[i][group] for i in range(3)]) for group in groups]
    print(mean, std, log_name)
    return {'mean': mean, 'std': std, 'log_name': log_name}



if __name__ == "__main__":
    plot_selection = [1, 1, 1, 1]

    base = [
        "crew_algorithms/ddpg/logs/0228_2008_rl.json",
        "crew_algorithms/ddpg/logs/0228_2008_rl.json",
        "crew_algorithms/ddpg/logs/0228_2008_rl.json",
    ]

    deploy_history_hf = [
        "crew_algorithms/ddpg/logs/0306_0524_hf.json",
        "crew_algorithms/ddpg/logs/0306_0524_hf.json",
        "crew_algorithms/ddpg/logs/0306_0524_hf.json",
    ]

    deploy_history = [
        "crew_algorithms/ddpg/logs/0306_0508_rl.json", 
        "crew_algorithms/ddpg/logs/0306_0508_rl.json", 
        "crew_algorithms/ddpg/logs/0306_0508_rl.json", 
    ]

    deploy = [
        "crew_algorithms/ddpg/logs/0306_2025_rl.json", 
        "crew_algorithms/ddpg/logs/0306_2025_rl.json", 
        "crew_algorithms/ddpg/logs/0306_2025_rl.json", 
    ]

    deploy_hf = [
        "crew_algorithms/ddpg/logs/0306_2028_hf.json",
        "crew_algorithms/ddpg/logs/0306_2028_hf.json",
        "crew_algorithms/ddpg/logs/0306_2028_hf.json",
    ]

    heuristic = [
        "crew_algorithms/ddpg/logs/0228_2009_rl.json", # heuristic
        "crew_algorithms/ddpg/logs/0228_2009_rl.json",
        "crew_algorithms/ddpg/logs/0228_2009_rl.json",
    ]


    
    heuristic_data = parse_logs_for_sr(heuristic, "heuristic")
    deploy_history_hf_data = parse_logs_for_sr(deploy_history_hf, "sparse feedback + learned feedback (long history) (human)")
    deploy_history_data = parse_logs_for_sr(deploy_history, "sparse feedback + learned feedback (long history)")
    deploy_data = parse_logs_for_sr(deploy, "sparse feedback + learned feedback")
    deploy_hf_data = parse_logs_for_sr(deploy_hf, "sparse feedback + learned feedback (human)")
    base_data = parse_logs_for_sr(base, "base")


    means = [base_data['mean'], deploy_data['mean'], deploy_hf_data['mean'], heuristic_data['mean']]
    stds = [base_data['std'], deploy_data['std'], deploy_hf_data['std'], heuristic_data['std']]
    log_names = [base_data['log_name'], deploy_data['log_name'], deploy_hf_data['log_name'], heuristic_data['log_name']]
    colors =  ['lightsteelblue', 'navajowhite', 'mediumaquamarine', 'lightcoral']
    means = [mean for i, mean in enumerate(means) if plot_selection[i]]
    stds = [std for i, std in enumerate(stds) if plot_selection[i]]
    log_names = [log_name for i, log_name in enumerate(log_names) if plot_selection[i]]
    colors = [color for i, color in enumerate(colors) if plot_selection[i]]
    make_bar_plot(means, stds, log_names, colors, save_img_path='307_bars_simple', groups=[40, 60, 80])

    
    heuristic_data = parse_logs_for_time(heuristic, "heuristic")
    deploy_history_hf_data = parse_logs_for_time(deploy_history_hf, "sparse feedback + learned feedback (long history) (human)")
    deploy_history_data = parse_logs_for_time(deploy_history, "sparse feedback + learned feedback (long history)")
    deploy_data = parse_logs_for_time(deploy, "sparse feedback + learned feedback")
    deploy_hf_data = parse_logs_for_time(deploy_hf, "sparse feedback + learned feedback (human)")
    base_data = parse_logs_for_time(base, "base")


    means = [base_data['mean'], deploy_data['mean'], deploy_hf_data['mean'], heuristic_data['mean']]
    stds = [base_data['std'], deploy_data['std'], deploy_hf_data['std'], heuristic_data['std']]
    log_names = [base_data['log_name'], deploy_data['log_name'], deploy_hf_data['log_name'], heuristic_data['log_name']]
    colors =  ['lightsteelblue', 'navajowhite', 'mediumaquamarine', 'lightcoral']
    means = [mean for i, mean in enumerate(means) if plot_selection[i]]
    stds = [std for i, std in enumerate(stds) if plot_selection[i]]
    log_names = [log_name for i, log_name in enumerate(log_names) if plot_selection[i]]
    colors = [color for i, color in enumerate(colors) if plot_selection[i]]
    make_bar_plot(means, stds, log_names, colors, save_img_path='307_bars_simple_time', mode='time', groups=[30, 60, 90])

    




    # plot_selection = [1, 0, 1, 0, 1]

    # base = [
    #     "crew_algorithms/ddpg/logs/0221_1108_rl.json",
    #     "crew_algorithms/ddpg/logs/0224_1140_rl_base_seed1.json",
    #     "crew_algorithms/ddpg/logs/0224_1140_rl.json",
    # ]

    # nodeploy = [
    #     "crew_algorithms/ddpg/logs/0222_0619_rl.json", # no aux
    #     "crew_algorithms/ddpg/logs/0223_1630_rl.json",
    #     "crew_algorithms/ddpg/logs/0223_2240_rl.json",
    # ]

    # deploy = [
    #     "crew_algorithms/ddpg/logs/0220_2354_rl.json", 
    #     "crew_algorithms/ddpg/logs/0223_2248_rl.json",
    #     "crew_algorithms/ddpg/logs/0224_0157_rl.json",
    # ]

    # aux = [
    #     "crew_algorithms/ddpg/logs/0224_1618_rl.json", # aux
    #     "crew_algorithms/ddpg/logs/0224_1639_rl.json",
    #     "crew_algorithms/ddpg/logs/0224_1640_rl.json",
    # ]

    # heuristic = [
    #     "crew_algorithms/ddpg/logs/heuristic_step_sr.json", # heuristic
    #     "crew_algorithms/ddpg/logs/0223_1623_rl.json",
    #     "crew_algorithms/ddpg/logs/0223_1625_rl.json",
    # ]

    
    # heuristic = parse_logs_for_sr(heuristic, "heuristic")
    # nodeploy = parse_logs_for_sr(nodeploy, "sparse feedback")
    # aux = parse_logs_for_sr(aux, "auxiliary loss")
    # deploy = parse_logs_for_sr(deploy, "sparse feedback + learned feedback")
    # base = parse_logs_for_sr(base, "base")


    # means = [base['mean'], nodeploy['mean'], deploy['mean'], aux['mean'], heuristic['mean']]
    # stds = [base['std'], nodeploy['std'], deploy['std'], aux['std'], heuristic['std']]
    # log_names = [base['log_name'], nodeploy['log_name'], deploy['log_name'], aux['log_name'], heuristic['log_name']]
    # colors =  ['lightsteelblue', 'navajowhite', 'mediumaquamarine', 'mediumpurple', 'lightcoral']
    # means = [mean for i, mean in enumerate(means) if plot_selection[i]]
    # stds = [std for i, std in enumerate(stds) if plot_selection[i]]
    # log_names = [log_name for i, log_name in enumerate(log_names) if plot_selection[i]]
    # colors = [color for i, color in enumerate(colors) if plot_selection[i]]
    # make_bar_plot(means, stds, log_names, colors, save_img_path='28_bars_simple')

    # base = [
    #     "crew_algorithms/ddpg/logs/0221_1108_rl.json",
    #     "crew_algorithms/ddpg/logs/0224_1140_rl_base_seed1.json",
    #     "crew_algorithms/ddpg/logs/0224_1140_rl.json",
    # ]

    # nodeploy = [
    #     "crew_algorithms/ddpg/logs/0222_0619_rl.json", # no aux
    #     "crew_algorithms/ddpg/logs/0223_1630_rl.json",
    #     "crew_algorithms/ddpg/logs/0223_2240_rl.json",
    # ]

    # deploy = [
    #     "crew_algorithms/ddpg/logs/0220_2354_rl.json", 
    #     "crew_algorithms/ddpg/logs/0223_2248_rl.json",
    #     "crew_algorithms/ddpg/logs/0224_0157_rl.json",
    # ]

    # aux = [
    #     "crew_algorithms/ddpg/logs/0224_1618_rl.json", # aux
    #     "crew_algorithms/ddpg/logs/0224_1639_rl.json",
    #     "crew_algorithms/ddpg/logs/0224_1640_rl.json",
    # ]

    # heuristic = [
    #     "crew_algorithms/ddpg/logs/heuristic_step_sr.json", # heuristic
    #     "crew_algorithms/ddpg/logs/0223_1623_rl.json",
    #     "crew_algorithms/ddpg/logs/0223_1625_rl.json",
    # ]


    
    # heuristic = parse_logs_for_time(heuristic, "heuristic")
    # nodeploy = parse_logs_for_time(nodeploy, "sparse feedback")
    # aux = parse_logs_for_time(aux, "auxiliary loss")
    # deploy = parse_logs_for_time(deploy, "sparse feedback + learned feedback")
    # base = parse_logs_for_time(base, "base")


    # means = [base['mean'], nodeploy['mean'], deploy['mean'], aux['mean'], heuristic['mean']]
    # stds = [base['std'], nodeploy['std'], deploy['std'], aux['std'], heuristic['std']]
    # log_names = [base['log_name'], nodeploy['log_name'], deploy['log_name'], aux['log_name'], heuristic['log_name']]
    # colors =  ['lightsteelblue', 'navajowhite', 'mediumaquamarine', 'mediumpurple', 'lightcoral']
    # means = [mean for i, mean in enumerate(means) if plot_selection[i]]
    # stds = [std for i, std in enumerate(stds) if plot_selection[i]]
    # log_names = [log_name for i, log_name in enumerate(log_names) if plot_selection[i]]
    # colors = [color for i, color in enumerate(colors) if plot_selection[i]]
    # make_bar_plot(means, stds, log_names, colors, save_img_path='28_bars_simple', mode='time', groups=[30, 60, 90])

    





    # log_files = [
        
    #     # 'crew_algorithms/ddpg/logs/0219_2159_rl.json',
    #     # 'crew_algorithms/ddpg/logs/0220_0158_rl.json',
    #     # "crew_algorithms/ddpg/logs/0220_1026_rl.json"
    #     # "crew_algorithms/ddpg/logs/0221_0623_rl.json",
    #     "crew_algorithms/ddpg/logs/heuristic_step_sr.json", # heuristic
    #     # "crew_algorithms/ddpg/logs/0220_1925_rl.json",
    #     # "crew_algorithms/ddpg/logs/0220_2032_rl.json",
    #     # "crew_algorithms/ddpg/logs/0220_2313_rl.json",
    #     # "crew_algorithms/ddpg/logs/0220_2354_rl.json", # by step
    #     "crew_algorithms/ddpg/logs/0221_1705_rl.json", # by traj
    #     "crew_algorithms/ddpg/logs/0222_0619_rl.json", # not deployed
    #     "crew_algorithms/ddpg/logs/0221_1108_rl.json", # baseline

    # ]
    # log_names = [
    #     # 'old',
        
    #     "heuristic",
    #     'learned feedaback deployed',
    #     'learned feedback not deployed',
    #     "baseline",
    #     # "split by step",
    #     # "split by traj"
        
    # ]
    
    # # make_plots(log_files, log_names, save_img_path='22')
    # make_bar_chart(log_files, log_names, save_img_path='22_bars')