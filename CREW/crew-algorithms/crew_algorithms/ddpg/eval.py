from time import time
import pickle
import json

import hydra
from attrs import define
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torchrl.trainers.helpers.collectors import (  
    OffPolicyCollectorConfig,
)

from crew_algorithms.envs.configs import EnvironmentConfig, register_env_configs
from crew_algorithms.ddpg.config import NetworkConfig, OptimizationConfig
from crew_algorithms.utils.wandb_utils import WandbConfig
from torchrl.envs.utils import ExplorationType, set_exploration_type

@define(auto_attribs=True)
class Config:
    envs: EnvironmentConfig = MISSING
    """Settings for the environment to use."""
    optimization: OptimizationConfig = OptimizationConfig()
    network: NetworkConfig = NetworkConfig()
    collector: OffPolicyCollectorConfig = OffPolicyCollectorConfig(
        frames_per_batch=1, init_random_frames=0
    )
    """Settings to use for the off-policy collector."""
    wandb: WandbConfig = WandbConfig(entity="lingyu-zhang", project="crew")
    """WandB logger configuration."""
    num_envs: int = 1
    seed: int = 42
    from_states: bool = False
    feedback_model: bool = False
    exp_path: str='none'
    eval_iter: int = 6

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
register_env_configs()

@hydra.main(version_base=None, config_path="../conf", config_name="ddpg")
def eval(cfg: Config):

    import os
    import uuid
    from collections import deque, defaultdict

    import torch
    import random
    import numpy as np

    from crew_algorithms.envs.channels import WrittenFeedbackChannel

    from crew_algorithms.ddpg.utils import (
        make_agent,
        make_env,
    )
    from crew_algorithms.utils.rl_utils import make_collector

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    cfg.envs.seed = cfg.seed

    written_feedback_queue = deque()
    with open('written_feedback_queue.pkl', 'wb') as f:
        pickle.dump(written_feedback_queue, f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Device:', device)

    def append_feedback(w):
        with open('written_feedback_queue.pkl', 'rb') as f:
            written_feedback_queue = pickle.load(f)
        written_feedback_queue.append(w)
        with open('written_feedback_queue.pkl', 'wb') as f:
            pickle.dump(written_feedback_queue, f)

    written_feedback_channel = WrittenFeedbackChannel(
        uuid.uuid4(),
        append_feedback,
    )

    env_fn = lambda: make_env(cfg.envs, written_feedback_channel, device)
    env = env_fn()

    model, actor, _ = make_agent(env, cfg, device)
    env.close()

    eval_envs = 1 if cfg.envs.name == 'bowling' else 100

    path = f'../Data/{cfg.exp_path}/'
    scores = defaultdict(int)

    for m in range(0, cfg.eval_iter):
        scores[m] = 0
        with set_exploration_type(ExplorationType.MODE), torch.no_grad():
            weights = torch.load(path + "%d.pth" %(m))
            model.load_state_dict(weights)

            collector = make_collector(cfg.collector, env_fn, model[0], device, cfg.num_envs)
            collector.set_seed(cfg.seed)
            for data in collector:
                if data["collector", "traj_ids"].max() >= eval_envs:
                    # print(scores)
                    with open(path + f"results-{cfg.seed}.json", "w") as f:
                        json.dump(scores, f)
                    collector.shutdown()
                    break
                score = data[("next", "agents", "reward")]
                # num_traj = data["collector", "traj_ids"].max() + 1
                # print(score,

                if score != 0:
                    scores[m] += score.item()
                    print(scores)
                    with open(path + f"results-{cfg.seed}.json", "w") as f:
                        json.dump(scores, f)
                    if cfg.envs.name == 'bowling':
                        collector.shutdown()
                        break
                # batch_num_success = int(data["next", "agents", "reward"].sum().item())
                # num_traj = data["next", "agents", "done"].sum().item()
                
                # print(batch_num_success , num_traj,  batch_num_success/num_traj)
                # scores[m] = batch_num_success/num_traj
                # # with open(path + f"results-{cfg.seed}.json", "w") as f:
                # #     json.dump(scores, f)
                # # print(scores[m])
                # # if i == 119:
                # #     num_traj = data["collector", "traj_ids"].max() + 1
                # #     scores[m] = scores[m] / num_traj
                # #     print(scores)
                # collector.shutdown()
                # break
                # i += 1
            # print("Collector shutdown")

if __name__ == "__main__":    
    eval()
