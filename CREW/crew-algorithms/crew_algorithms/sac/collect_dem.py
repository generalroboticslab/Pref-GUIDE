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
from crew_algorithms.sac.audio_feedback import Audio_Streamer
from crew_algorithms.sac.config import NetworkConfig, OptimizationConfig
from crew_algorithms.utils.wandb_utils import WandbConfig


@define(auto_attribs=True)
class Config:
    envs: EnvironmentConfig = MISSING
    """Settings for the environment to use."""
    optimization: OptimizationConfig = OptimizationConfig()
    network: NetworkConfig = NetworkConfig()
    collector: OffPolicyCollectorConfig = OffPolicyCollectorConfig(
        frames_per_batch=1, init_random_frames=100
    )
    """Settings to use for the off-policy collector."""
    wandb: WandbConfig = WandbConfig(entity="lingyuz", project="sac")
    """WandB logger configuration."""
    batch_size: int = 1
    buffer_size: int = 1_000_000
    audio: bool = False
    traj: bool = False


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
register_env_configs()


@hydra.main(version_base=None, config_path="../conf", config_name="sac")
def sac(cfg: Config):
    import os
    import uuid
    from collections import deque

    import torch
    import wandb
    from sortedcontainers import SortedList
    from torchrl.record.loggers import generate_exp_name, get_logger

    from crew_algorithms.envs.channels import WrittenFeedbackChannel
    from crew_algorithms.sac.trajectory_feedback import (
        TrajectoryFeedback,
    )
    from crew_algorithms.sac.utils import (
        audio_feedback,
        build_traj_id_to_ranking_map,
        combine_feedback_and_rewards,
        gradient_weighted_average_transform,
        human_delay_transform,
        make_agent,
        make_data_buffer,
        make_env,
        make_loss_module,
        make_optimizer,
        override_il_feedback,
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

    ranked_trajectories = SortedList()
    written_feedback_queue = deque()
    with open('written_feedback_queue.pkl', 'wb') as f:
        pickle.dump(written_feedback_queue, f)


    device = "cpu" if not torch.has_cuda else "cuda:0"

    def append_feedback(w):
        with open('written_feedback_queue.pkl', 'rb') as f:
            written_feedback_queue = pickle.load(f)
        written_feedback_queue.append(w)
        with open('written_feedback_queue.pkl', 'wb') as f:
            pickle.dump(written_feedback_queue, f)
        # print(written_feedback_queue, len(written_feedback_queue))

    written_feedback_channel = WrittenFeedbackChannel(
        uuid.uuid4(),
        append_feedback,
        # lambda written_feedback: written_feedback_queue.append(written_feedback),
    )

    env_fn = lambda: make_env(cfg.envs, written_feedback_channel, device)
    # env_fn = make_env(cfg.envs, written_feedback_channel, device)
    env = env_fn()

    num_actions = env.action_spec.space.n
    model, actor = make_agent(env, device, num_actions)
    model.eval()
    actor.eval()

    loss_module, target_net_updater = make_loss_module(cfg, env, model, num_actions)

    env.close()

    collector = make_collector(cfg.collector, env_fn, actor, device)

    collected_frames = 0
    human_delay_buffer_td = None

    global_start_time = time()
    collected = 0

    from torchvision.utils import save_image
    tic = time()

    for i, data in enumerate(collector):
        data = data.view(-1)

        traj_num = data["agents", "observation", "obs_0_1"][...,3]
        if (i + 1) % 10 == 0:
            print('Traj: %d| Collected: %d ' % (traj_num.item(), collected), end='\r')
        # print(data["agents", "action"])

        if data["agents", "observation", "obs_0_1"][..., 1] == 1:
            act = data["agents", "observation", "obs_0_1"][..., 2].item()
            obs = data["agents", "observation", "obs_0_0"].squeeze(1)
            # vec = data["agents", "encoder_vec"]
            # print(act, vec.shape)

            torch.save(act, '../Data/BC_data/act/a_%d.pt'%(i))
            save_image(obs, '../Data/BC_data/obs/o_%d.png'%(i))
            collected += 1

    print("end- ", round(time() - global_start_time, 4))


if __name__ == "__main__":
    sac()