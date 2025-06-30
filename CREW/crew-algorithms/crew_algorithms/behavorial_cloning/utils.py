# import copy
import math
import pdb
import torch
from tensordict.nn import InteractionType
from torch import nn, optim
from torch.distributions import Categorical
from torchrl.data import LazyMemmapStorage, TensorDictPrioritizedReplayBuffer, ListStorage, LazyTensorStorage
from torchrl.envs import Compose, StepCounter, ToTensorImage, TransformedEnv
from torchrl.modules import ProbabilisticActor, SafeModule, ValueOperator
from torchrl.objectives import DiscreteSACLoss, SoftUpdate
from torchvision import transforms

from crew_algorithms.auto_encoder import EncoderTransform
from crew_algorithms.envs.channels import WrittenFeedbackChannel
from crew_algorithms.envs.configs import EnvironmentConfig
from crew_algorithms.bc_pretrain.imitation_learning import (
    ImitationLearningWrapper,
)
from crew_algorithms.bc_pretrain.policy import ActorNet, QValueNet
from crew_algorithms.utils.rl_utils import make_base_env

# from multiprocessing import Manager
# from multiprocessing.managers import BaseManager


def make_env(
    cfg: EnvironmentConfig,
    written_feedback_channel: WrittenFeedbackChannel,
    device: str,
):
    """Creates an environment based on the configuration that can be used for
    a random policy.

    Args:
        cfg: The environment configuration to be used.
        written_feedback_channel: A Unity side channel that can be used
            to share written feedback at the end of each episode.
        device: The device to perform environment operations on.

    Returns:
        The environment that can be used for the random policy.
    """

    # env = make_base_env(cfg, device,
    # written_feedback_channel=written_feedback_channel)

    # TODO: check encoder_transform
    env = TransformedEnv(
        make_base_env(cfg, device, written_feedback_channel=written_feedback_channel),
        Compose(
            ToTensorImage(
                in_keys=[("agents", "observation", "obs_0_0")], unsqueeze=True
            ),
            EncoderTransform(
                env_name=cfg.name,
                num_channels=cfg.num_stacks * cfg.num_channels,
                in_keys=[("agents", "observation", "obs_0_0")],
                out_keys=[(("agents", "encoder_vec"))],
                version="latest",
            ),
            StepCounter(),
        ),
    )
    return env


def make_agent(proof_env, device, num_actions):
    # Define Actor Network
    actor_net = ActorNet(
        n_agent_inputs=proof_env.observation_spec["agents", "encoder_vec"].shape[-1],
        num_cells=[256, 256],
        n_agent_outputs=num_actions,
        activation_class=nn.ReLU,
    )

    actor_module = SafeModule(
        actor_net,
        in_keys=[("agents", "encoder_vec")],
        out_keys=[
            ("agents", "logits"),
        ],
    )
    actor = ProbabilisticActor(
        spec=proof_env.action_spec,
        in_keys=[("agents", "logits")],
        out_keys=[proof_env.action_key],
        module=actor_module,
        distribution_class=Categorical,
        # distribution_class=lambda logits: Categorical(logits=logits),
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )
    actor = ImitationLearningWrapper(
        actor,
        action_key=proof_env.action_key,
        il_enabled_key=("agents", "observation", "obs_0_1"),
        il_action_key=("agents", "observation", "obs_0_1"),
    )

    # Define Critic Network
    qvalue_net = QValueNet(
        n_agent_inputs=proof_env.observation_spec["agents", "encoder_vec"].shape[-1],
        num_cells=[256, 256],
        n_agent_outputs=num_actions,
        activation_class=nn.ReLU,
    )
    qvalue = ValueOperator(
        in_keys=[("agents", "encoder_vec")],
        out_keys=[("agents", "action_value")],
        module=qvalue_net,
    )

    model = torch.nn.ModuleList([actor, qvalue]).to(device)

    return model, model[0]


def make_data_buffer(cfg):
    # BaseManager.register('TensorDictPrioritizedReplayBuffer',
    #  TensorDictPrioritizedReplayBuffer)
    # manager = BaseManager()
    # manager.start()
    replay_buffer = TensorDictPrioritizedReplayBuffer(
        alpha=0.7,
        beta=0.5,
        pin_memory=False,
        storage=LazyMemmapStorage(cfg.buffer_size),
        batch_size=cfg.batch_size,
        priority_key="rank_adjusted_td_error",
    )

    return replay_buffer


def make_loss_module(cfg, env, model, num_actions):
    """Make loss module and target network updater."""
    # Create SAC loss
    loss_module = DiscreteSACLoss(
        actor_network=model[0].td_module,
        action_space=env.action_spec,
        qvalue_network=model[1],
        num_actions=num_actions,
        num_qvalue_nets=2,
        target_entropy_weight=cfg.optimization.target_entropy_weight,
        loss_function="smooth_l1",
    )
    loss_module.set_keys(
        action=env.action_key,
        reward=env.reward_key,
        done=env.done_key,
        value=("agents", "state_value"),
        action_value=("agents", "action_value"),
        priority=("agents", "td_error"),
    )
    loss_module.make_value_estimator(gamma=cfg.optimization.gamma)
    return loss_module


def make_optimizer(cfg, loss_module):
    # Define Target Network Updater
    target_net_updater = SoftUpdate(
        loss_module, eps=cfg.optimization.target_update_polyak
    )

    # Define optimizer
    optimizer = optim.Adam(
        loss_module.parameters(),
        lr=cfg.optimization.lr,
        weight_decay=cfg.optimization.weight_decay,
    )
    return target_net_updater, optimizer


def build_traj_id_to_ranking_map(trajectory_feedbacks):
    traj_id_to_ranking = {
        trajectory_feedback.id: i + 1
        for i, trajectory_feedback in enumerate(trajectory_feedbacks)
    }

    # Set one extra trajectory feedback for the current trajectory.
    # Make the current trajectory have the worst ranking so it is prioritized.
    traj_id_to_ranking[len(traj_id_to_ranking)] = len(trajectory_feedbacks) + 1
    return traj_id_to_ranking


def human_delay_transform(td, shift_key, N):
    data = td.get(shift_key)
    human_feedback = data[..., 0]

    # human_feedback_0 = (
    #     torch.ones((N, *human_feedback.shape[1:]), device=human_feedback.device) * -9
    # )

    human_feedback_0 = (
        torch.ones((N, *human_feedback.shape[1:]), device=human_feedback.device) * 0
    )

    human_feedback = torch.cat([human_feedback, human_feedback_0], 0)
    human_feedback = human_feedback[N:]

    data[..., 0] = human_feedback
    td.set(shift_key, data)
    # Return the tensors with valid data and the tensors with placeholder separately
    return td[:-N], td[-N:]


def gradient_weighted_average_transform(td, key, N, step_count_key="step_count"):
    # Cannot compute gradients and average on 1 length tensor
    if td.batch_size[0] <= 1:
        return td

    data = td.get(key)
    step_count = td[step_count_key].squeeze()
    feedback = data[..., 0].squeeze()

    coordinates = (step_count,)
    try:
        grads = torch.gradient(feedback, spacing=coordinates)[0]
    except Exception as e:
        print(step_count.shape)
        print(step_count)
        print(td)
        raise e

    total_padding = N - 1

    front_padding = torch.zeros(math.floor(total_padding / 2), device=feedback.device)
    back_padding = torch.zeros(math.ceil(total_padding / 2), device=feedback.device)

    feedback = torch.cat([front_padding, feedback, back_padding], -1)
    grads = torch.cat([front_padding, grads, back_padding], -1)

    feedback = feedback.unfold(-1, N, 1)
    grads = grads.unfold(-1, N, 1)

    weighted_average = torch.sum(feedback * grads, dim=-1) / (grads.sum(dim=-1) + 1e-10)
    data[..., 0] = torch.reshape(weighted_average, data[..., 0].shape)
    td.set(key, data)
    return td


def override_il_feedback(td, il_enabled_key, feedback_key, il_feedback):
    # https://arxiv.org/pdf/1905.06750.pdf
    # https://arxiv.org/pdf/2108.04763.pdf
    original_feedback = td.get(feedback_key)
    feedback = original_feedback[..., 0]

    # print(td.get(il_enabled_key))
    il_enabled = td.get(il_enabled_key)[..., 1].bool()
    # print('IL on:', il_enabled.sum())
    feedback[il_enabled] = il_feedback
    original_feedback[..., 0] = feedback
    td.set(feedback_key, original_feedback)
    return td

def combine_feedback_and_rewards(td, feedback_key, reward_key):
    print((td.get(feedback_key)[..., 0].unsqueeze(dim=-1) * 0.05).sum())
    td[reward_key] = td[reward_key] + td.get(feedback_key)[..., 0].unsqueeze(dim=-1) * 0.05
    return td
