from time import time
import pickle

import hydra
from attrs import define
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torchrl.trainers.helpers.collectors import (  # make_collector_offpolicy,
    OffPolicyCollectorConfig,
)

from crew_algorithms.envs.configs import EnvironmentConfig, register_env_configs
from crew_algorithms.bc_pretrain.config import NetworkConfig, OptimizationConfig
from crew_algorithms.utils.wandb_utils import WandbConfig


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
    wandb: WandbConfig = WandbConfig(entity="lingyuz", project="bc_pretrain")
    """WandB logger configuration."""
    batch_size: int = 1
    buffer_size: int = 1_800_000 # 10hours of game 100_000_000

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
register_env_configs()


@hydra.main(version_base=None, config_path="../conf", config_name="bc_pretrain")
def behavioral_cloning(cfg: Config):
    import os

    import torch
    import wandb
    from sortedcontainers import SortedList
    from torchrl.record.loggers import generate_exp_name, get_logger

    from crew_algorithms.bc_pretrain.utils import (
        make_agent,
        make_env,
    )
    from crew_algorithms.utils.rl_utils import make_collector

    wandb.login()

    exp_name = generate_exp_name(
        "MultiModalFeedback", f"multimodal-feedback-{cfg.envs.name}"
    )
    logger = get_logger(
        "wandb",
        logger_name=os.getcwd(),
        experiment_name=exp_name,
        wandb_kwargs=dict(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            settings=wandb.Settings(start_method="thread"),
            tags=[cfg.envs.name],
        ),
    )
    logger.log_hparams(cfg)

    device = "cpu" if not torch.has_cuda else "cuda:0"

    env_fn = lambda: make_env(cfg.envs, None, device)
    env = env_fn()

    num_actions = env.action_spec.space.n
    _, actor = make_agent(env, device, num_actions)

    env.close()

    collector = make_collector(cfg.collector, env_fn, actor, device)

    global_start_time = time()

    for i, data in enumerate(collector):
        traj_num = data["agents", "observation", "obs_0_1"][...,3]
        if (i + 1) % 10 == 0:
            print(traj_num)

        if traj_num > 500:
            break

        data = data.view(-1)

        pos = data["agents", "observation", "obs_0_1"][...,-6:]
        vec = data["agents", "encoder_vec"]

        torch.save(pos, '/home/lingyu/Desktop/lingyu/CREW/Pretrain_Dataset/pos2/%d.pt'%i)
        torch.save(vec, '/home/lingyu/Desktop/lingyu/CREW/Pretrain_Dataset/vec2/%d.pt'%i)

    print("end- ", round(time() - global_start_time, 4))


if __name__ == "__main__":
    bc_pretrain()
