from collections import defaultdict, deque

import torch
from hydra.utils import instantiate
from tensordict.nn import TensorDictModule
from tensordict.tensordict import TensorDictBase
from torch.distributions import Categorical
from torchrl.data.replay_buffers import ListStorage, RandomSampler, ReplayBuffer
from torchrl.data.tensor_specs import DiscreteTensorSpec
from torchrl.envs import Compose, EnvBase, StepCounter, TransformedEnv
from torchrl.modules import ProbabilisticActor

from crew_algorithms.auto_encoder import EncoderTransform
from crew_algorithms.deep_coach.optim import DeepCoach, DeepCoachConfig
from crew_algorithms.deep_coach.policy import PolicyNet
from crew_algorithms.envs.channels import ToggleTimestepChannel
from crew_algorithms.envs.configs import EnvironmentConfig
from crew_algorithms.utils.rl_utils import make_base_env
from crew_algorithms.utils.transforms import CatUnitySensorsAlongChannelDimTransform


def make_env(
    cfg: EnvironmentConfig,
    toggle_timestep_channel: ToggleTimestepChannel,
    device: str,
):
    """Creates an environment based on the configuration that can be used for
    the DeepCoach algorithm.

    Args:
        cfg: The environment configuration to be used.
        toggle_timestep_channel: A Unity side channel that can be used
            to play/pause games.
        device: The device to perform environment operations on.

    Returns:
        The environment that can be used for DeepCoach.
    """
    env = TransformedEnv(
        make_base_env(cfg, device, toggle_timestep_channel=toggle_timestep_channel),
        Compose(
            CatUnitySensorsAlongChannelDimTransform(in_keys="observation"),
            EncoderTransform(
                env_name=cfg.name,
                num_channels=cfg.num_stacks * cfg.num_channels,
                in_keys="observation",
                version="latest",
            ),
            StepCounter(),
        ),
    )
    return env


def make_policy(env: EnvBase):
    """Creates the DeepCoach policy.

    Args:
        env: The environment that the policy will be used for.

    Returns:
        The DeepCoach policy.
    """
    if not isinstance(env.action_spec, DiscreteTensorSpec):
        raise TypeError("Only discrete action tensor specs can be used with DeepCoach")
    num_actions = env.action_spec.space.n
    policy_net = PolicyNet(num_actions)
    tensordict_module = TensorDictModule(
        policy_net, in_keys=["encoder_vec"], out_keys=["probs"]
    )
    policy = ProbabilisticActor(
        module=tensordict_module,
        spec=env.action_spec,
        in_keys=["probs"],
        distribution_class=lambda probs: Categorical(probs=probs),
        default_interaction_mode="mode",
        return_log_prob=True,
    )
    return policy


def make_data_buffer(cfg):
    """Makes the data storage elements required for the DeepCoach policy.

    Args:
        cfg: The DeepCoach configuration.

    Returns:
        A defaultdict of windows which is a mapping from agent ids to
            a deque of size window length.
        A sample storage which is used to store samples before feedback
            has been collected for them (as feedback is applied to samples
            that occurred human delay timesteps ago). It is a mapping
            from agent ids to a deque of size human delay.
        A replay buffer which can be used to store windows of experiences.
    """
    windows = defaultdict[str, deque[TensorDictBase]](
        lambda: deque(maxlen=cfg.window_length)
    )
    sample_storage = defaultdict[str, deque[TensorDictBase]](
        lambda: deque(maxlen=cfg.envs.human_delay)
    )
    replay_buffer = ReplayBuffer(
        storage=ListStorage(cfg.buffer_storage),
        sampler=RandomSampler(),
        collate_fn=lambda x: x,
        batch_size=cfg.mini_batch_size,
    )
    return windows, sample_storage, replay_buffer


def make_optim(cfg: DeepCoachConfig, policy: ProbabilisticActor):
    """Creates the DeepCoach optimizer.

    Args:
        cfg: The DeepCoach optimizer configuration.
        policy: The policy to be optimized.

    Returns:
        Optimizer defined by the DeepCoach algorithm.
    """
    optim: DeepCoach = instantiate(cfg)(policy.parameters())
    return optim


def add_samples_to_buffers(data_view, sample_storage, windows, replay_buffer, logger):
    """Adds samples collected from the collector to the relevant storage buffers.

    Stores the current data in the sample storage. Once a sample has occurred human
    delay timesteps ago, it moves the sample that occurred human delay timesteps ago
    from the sample storage to the current window of experiences and also moves
    the feedback collected now to the moved sample (the one that occurred human delay
    timesteps ago) and resets the current sample's feedback to 0.

    If the window is non-empty and the feedback of the moved sample is nonzero, it
    commits the window of experiences to the replay buffer.

    Args:
        data_view: The current data collected from the data collector.
        sample_storage: A defaultdict which is used to store samples before feedback
            has been collected for them (as feedback is applied to samples
            that occurred human delay timesteps ago). It is a mapping
            from agent ids to a deque of size human delay.
        windows: A defaultdict which is a mapping from agent ids to
            a deque of size window length.
        replay_buffer: A replay buffer which can be used to store windows of
            experiences.
        logger: A logger which can be used to log metrics.
    """
    for single_data_view in data_view.unbind(0):
        agent_id = single_data_view["agent_id"].item()
        if len(sample_storage[agent_id]) == sample_storage[agent_id].maxlen:
            # Store the sample that was occurred human_delay timesteps ago
            # in the window.
            windows[agent_id].append(sample_storage[agent_id].popleft())
        sample_storage[agent_id].append(single_data_view)

        if len(windows[agent_id]) > 0:
            # The sample that occurred d timesteps ago is the last element
            # of the window since we just added the current timestep sample
            # to the sample storage kicking the d timestep ago sample
            # to the window.
            target_sample = windows[agent_id][-1]

            # Move the feedback to human_delay timesteps back. Also clamp
            # it as the feedback is a sum and we want to make sure it is still
            # in the set {-1, 0, 1}.
            target_sample["feedback"] = single_data_view["feedback"].clamp(-1, 1)
            single_data_view["feedback"] = torch.zeros_like(
                single_data_view["feedback"]
            )

            logger.log_scalar(
                "feedback",
                target_sample["feedback"].item(),
                target_sample["step_count"].item(),
            )

            if target_sample["feedback"] != 0:
                replay_buffer.add(list(windows[agent_id]))
                windows[agent_id].clear()
