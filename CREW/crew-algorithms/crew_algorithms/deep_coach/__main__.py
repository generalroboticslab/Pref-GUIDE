import hydra
from attrs import define
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torchrl.trainers.helpers.collectors import OffPolicyCollectorConfig

from crew_algorithms.deep_coach.optim import DeepCoachConfig
from crew_algorithms.envs.configs import EnvironmentConfig, register_env_configs
from crew_algorithms.utils.wandb_utils import WandbConfig


@define(auto_attribs=True)
class Config:
    envs: EnvironmentConfig = MISSING
    """Settings for the environment to use."""
    collector: OffPolicyCollectorConfig = OffPolicyCollectorConfig(
        exploration_mode="mode", frames_per_batch=8, total_frames=1000
    )
    """Settings to use for the off-policy collector."""
    optim: DeepCoachConfig = DeepCoachConfig()
    """Settings for the optimizer defined by the DeepCoach algorithm."""
    wandb: WandbConfig = WandbConfig(project="deep-coach")
    """WandB logger configuration."""
    mini_batch_size: int = 16
    """Size of the mini batches stored in the Replay Buffer."""
    buffer_storage: int = 50
    """Maximium size of the Replay Buffer."""
    window_length: int = 2
    """Defines the maximum number (n - window_length) timesteps a feedback
    is applied to."""


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
register_env_configs()


@hydra.main(version_base=None, config_path="../conf", config_name="deep_coach")
def deep_coach(cfg: Config):
    """An implementation of the DeepCoach algorithm.

    For more details, see the DeepCoach paper: https://arxiv.org/pdf/1902.04257.pdf.
    """
    import os
    import uuid

    import torch
    import wandb
    from torchrl.record.loggers import generate_exp_name, get_logger

    from crew_algorithms.deep_coach.utils import (
        add_samples_to_buffers,
        make_data_buffer,
        make_env,
        make_optim,
        make_policy,
    )
    from crew_algorithms.envs.channels import ToggleTimestepChannel
    from crew_algorithms.utils.rl_utils import log_policy, make_collector

    wandb.login()
    exp_name = generate_exp_name("DeepCoach", f"deep-coach-{cfg.envs.name}")
    logger = get_logger(
        "wandb",
        logger_name=os.getcwd(),
        experiment_name=exp_name,
        wandb_kwargs=dict(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            settings=wandb.Settings(start_method="thread"),
            tags=["baseline", cfg.envs.name],
        ),
    )
    logger.log_hparams(cfg)

    device = "cpu" if not torch.has_cuda else "cuda:0"
    toggle_timestep_channel = ToggleTimestepChannel(uuid.uuid4())

    env = make_env(cfg.envs, toggle_timestep_channel, device)
    policy = make_policy(env)
    collector = make_collector(cfg.collector, env, policy, device)
    windows, sample_storage, replay_buffer = make_data_buffer(cfg)
    optim = make_optim(cfg.optim, policy)

    for data in collector:
        for single_data_view in data.unbind(0):
            logger.log_scalar(
                "sample_log_prob",
                single_data_view["sample_log_prob"].item(),
                single_data_view["step_count"].item(),
            )
            logger.log_scalar(
                "reward",
                single_data_view["reward"].item(),
                single_data_view["step_count"].item(),
            )

        add_samples_to_buffers(data, sample_storage, windows, replay_buffer, logger)

        if len(replay_buffer) > cfg.mini_batch_size:
            batch = replay_buffer.sample()
            for window in batch:
                optim.zero_grad()
                for sample in window:
                    # get_dist recomputes the action/probabilities
                    # based on the policy now, which is what we want.
                    current_distribution = policy.get_dist(sample.clone(recurse=False))
                    sample["sample_log_prob_now"] = current_distribution.log_prob(
                        sample["action"]
                    )
                optim.update_eligibility_trace(window)
            # Get the distribution at the current time step for entropy regularization.
            current_td = data.unbind(0)[-1]
            optim.zero_grad()
            optim.update_entropy_regularization(policy.get_dist(current_td))
            optim.zero_grad()
            optim.step()

    log_policy(cfg, policy, logger)


if __name__ == "__main__":
    deep_coach()
