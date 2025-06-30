import torch
from tensordict.nn import TensorDictModule
from tensordict.utils import NestedKey
from torchrl.data.replay_buffers import (
    LazyMemmapStorage,
    RandomSampler,
    TensorDictPrioritizedReplayBuffer,
)
from torchrl.envs import Compose, EnvBase, StepCounter, ToTensorImage, TransformedEnv

from crew_algorithms.auto_encoder import EncoderTransform
from crew_algorithms.envs.channels import ToggleTimestepChannel
from crew_algorithms.envs.configs import EnvironmentConfig
from crew_algorithms.hackathon_demo.imitation_learning import ImitationLearningWrapper
from crew_algorithms.hackathon_demo.policy import PolicyNet
from crew_algorithms.utils.rl_utils import make_base_env


def make_env(
    cfg: EnvironmentConfig,
    toggle_timestep_channel: ToggleTimestepChannel,
    device: str,
):
    """Creates an environment based on the configuration that can be used for
    a random policy.

    Args:
        cfg: The environment configuration to be used.
        toggle_timestep_channel: A Unity side channel that can be used
            to play/pause games.
        device: The device to perform environment operations on.

    Returns:
        The environment that can be used for the random policy.
    """
    env = TransformedEnv(
        make_base_env(cfg, device, toggle_timestep_channel=toggle_timestep_channel),
        Compose(
            StepCounter(),
        ),
    )
    return env


def make_policy(env: EnvBase):
    policy = ImitationLearningWrapper(
        TensorDictModule(
            PolicyNet(env.action_spec),
            in_keys=[("agents", "observation")],
            out_keys=env.action_key,
        ),
        action_spec=env.action_spec,
        action_key=env.action_key,
        il_enabled_key=("agents", "observation", "obs_0_1"),
        il_action_key=("agents", "observation", "obs_0_1"),
    )
    return policy


def make_data_buffer(cfg):
    replay_buffer = TensorDictPrioritizedReplayBuffer(
        storage=LazyMemmapStorage(cfg.buffer_storage),
        sampler=RandomSampler(),
        batch_size=cfg.mini_batch_size,
    )
    return replay_buffer


def count_anomalies_in_window(window, action_key: NestedKey):
    num_anomalies = 0
    prev_action = None
    for td in window:
        action = td[action_key]
        if action != prev_action:
            num_anomalies += 1
        prev_action = action
    return num_anomalies
