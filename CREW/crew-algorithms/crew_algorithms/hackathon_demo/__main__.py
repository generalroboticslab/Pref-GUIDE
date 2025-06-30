import hydra
from attrs import define
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torchrl.trainers.helpers.collectors import OnPolicyCollectorConfig

from crew_algorithms.envs.configs import EnvironmentConfig, register_env_configs
from crew_algorithms.utils.wandb_utils import WandbConfig


@define(auto_attribs=True)
class Config:
    envs: EnvironmentConfig = MISSING
    """Settings for the environment to use."""
    collector: OnPolicyCollectorConfig = OnPolicyCollectorConfig(frames_per_batch=1)
    """Settings to use for the on-policy collector."""
    wandb: WandbConfig = WandbConfig(project="hackathon_demo")
    """WandB logger configuration."""


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
register_env_configs()


@hydra.main(version_base=None, config_path="../conf", config_name="hackathon_demo")
def hackathon_demo(cfg: Config):
    """An implementation of a random policy."""
    import os
    import uuid

    import torch
    import wandb
    from torchrl.record.loggers import generate_exp_name, get_logger

    from crew_algorithms.envs.channels import ToggleTimestepChannel
    from crew_algorithms.hackathon_demo.utils import make_env, make_policy
    from crew_algorithms.utils.rl_utils import make_collector

    wandb.login()
    exp_name = generate_exp_name("HackathonDemo", f"hackathon-demo-{cfg.envs.name}")
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
    toggle_timestep_channel = ToggleTimestepChannel(uuid.uuid4())

    env = make_env(cfg.envs, toggle_timestep_channel, device)
    policy = make_policy(env)
    collector = make_collector(cfg.collector, env, policy, device)

    for batch, data in enumerate(collector):
        for single_data_view in data.unbind(0):
            # print(single_data_view["agents", "observation", "obs_0_1"][0][2])
            pass


if __name__ == "__main__":
    hackathon_demo()
