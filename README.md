# Pref-GUIDE: Continual Policy Learning from Real-Time Human Feedback via Preference-Based Rewards
[Zhengran Ji](https://jzr01.github.io/)¹, [Boyuan Chen](http://boyuanchen.com/)¹

¹ Duke University

[website](http://generalroboticslab.com/Pref-GUIDE) | [paper](link_here) | [video](video_link)

![Multi-Agent/Robot Collaboration](images/Teaser.jpeg)


## Overview
Humans play a crucial role in training reinforcement learning agents, particularly when designing explicit reward functions is difficult. Prior work has focused on either offline labeling of trajectories or real-time human feedback during the agent’s learning process. However, real-time feedback is often expensive. To reuse this valuable data beyond the human-in-the-loop phase, the state-of-the-art method, GUIDE, trained a regression-based reward model for continual learning. Nevertheless, human preferences are inherently inconsistent and noisy, leading to inaccurate reward predictions and suboptimal performance in continual learning. We propose Pref-GUIDE, a novel method that transforms real-time scalar human feedback collected from GUIDE into preference-based datasets for training preference-based reward models. Pref-GUIDE Individual addresses temporal inconsistency through a moving window sampling strategy and accounts for inherent human randomness by introducing a no-preference range. To further enhance robustness, we leverage feedback from 50 human subjects by introducing a voting-based relabeling scheme, Pref-GUIDE Voting, that aggregates multiple subject-specific reward models to generate consensus preference labels. Experiments across three distinct environments demonstrate that Pref-GUIDE Individual outperforms GUIDE during continual learning. Furthermore, Pref-GUIDE Voting achieves even greater performance, surpassing both Pref-GUIDE Individual and human-designed heuristic dense rewards in complex environments. Our study proposes a novel method for reusing valuable real-time scalar human feedback by converting it into preference-based data for training more robust and consistent reward models for continual learning after the human left loop..

![Method](images/Mainfig.jpeg)

## Result
![Method](images/Simulation.png)



## Quick Start

1. Clone the repository:

    ```bash
    git clone https://github.com/generalroboticslab/HUMAC.git
    ```
    
## Repository Structure
This repository has the following structure
```plaintext
├── Simulation              
│   └── crew-algorithms
│   └── environment
│   └── training
├── Real-World
│   └── environment
│   └── training
├── images
├── .gitignore              
├── README.md           
└── LICENSE             

```

## Acknowledgement


This work is supported by the ARL STRONG program under awards W911NF2320182 and W911NF2220113. We also thank [Lingyu Zhang](https://www.jiaxunliu.com/) for helpful discussion.


## Citation

If you think this paper is helpful, please consider citing our work

```plaintext
Add Citation here
```

