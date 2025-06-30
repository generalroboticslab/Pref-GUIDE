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
from crew_algorithms.sac.config import NetworkConfig, OptimizationConfig
from crew_algorithms.utils.wandb_utils import WandbConfig

import torch

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

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
register_env_configs()

@hydra.main(version_base=None, config_path="../conf", config_name="sac")
def eval(cfg: Config):

    import os
    import uuid
    from collections import deque, defaultdict

    import torch
    import random
    import numpy as np

    from crew_algorithms.envs.channels import WrittenFeedbackChannel

    from crew_algorithms.sac.utils import (
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

    eval_envs = 1 if cfg.envs.name == 'bowling' else 11

    path = f'../Data/FINAL/{cfg.exp_path}/'
    scores = defaultdict(int)
    explore = {}

    for m in range(0, 6):
        scores[m] = 0
        explore[m] = {}

        weights = torch.load(path + "%d.pth" %(m))
        model.load_state_dict(weights)

        collector = make_collector(cfg.collector, env_fn, model[0], device, cfg.num_envs)
        collector.set_seed(cfg.seed)
        for data in collector:
            this_id = data["collector", "traj_ids"].item()

            try:
                explore[m][this_id].append(count_non_black_pixel(data))
            except:
                explore[m][this_id]= [count_non_black_pixel(data)]
            
            print(explore)
            
            if data["collector", "traj_ids"].max() >= eval_envs:
                print(scores)
                with open(path + f"results.json", "w") as f:
                    json.dump(scores, f)
                collector.shutdown()
                break
            score = data[("next", "agents", "reward")]
            if score != 0:
                scores[m] += score.item()
                print(scores)
                with open(path + f"results.json", "w") as f:
                    json.dump(scores, f)

def count_non_black_pixel(data):
    
    return torch.sum(torch.sum(data['agents','observation','obs_0_0'][:,:,-3:,:,:].squeeze(),dim=0) != 0).item()


if __name__ == "__main__":    
    eval()
