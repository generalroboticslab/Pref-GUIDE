# import copy
import os
from datetime import datetime
import math
import pdb
import torch
from tensordict.nn import InteractionType
from tensordict.nn import (
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
    TensorDictSequential,
    )
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules.distributions import TanhNormal
from torch import nn, optim
from torch.distributions import Categorical
# from torchrl.data import LazyMemmapStorage, TensorDictPrioritizedReplayBuffer, ListStorage, LazyTensorStorage\
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import PrioritizedSampler, RandomSampler
from torchrl.envs.transforms.transforms import ObservationTransform, Compose, CenterCrop, Resize, StepCounter, ToTensorImage, TransformedEnv, FrameSkipTransform, ObservationNorm, _apply_to_composite, CatFrames, UnsqueezeTransform
from torchrl.modules import ProbabilisticActor, SafeModule, ValueOperator
from torchrl.objectives import DiscreteSACLoss, SoftUpdate, SACLoss

from crew_algorithms.auto_encoder import EncoderTransform
from crew_algorithms.auto_encoder.model import Encoder, StateEncoder
from crew_algorithms.envs.channels import WrittenFeedbackChannel
from crew_algorithms.envs.configs import EnvironmentConfig
from crew_algorithms.sac.audio_feedback import audio2reward
from crew_algorithms.sac.imitation_learning import (
    ImitationLearningWrapper,
)
from crew_algorithms.sac.policy import ActorNet, QValueNet, ContinuousActorNet, ContinuousQValueNet, FeedbackNet
from crew_algorithms.utils.rl_utils import make_base_env

from torchvision.transforms import RandomAffine, RandomRotation, InterpolationMode
from torchrl.envs.transforms.utils import _set_missing_tolerance
from torchvision import transforms
from torchvision.utils import save_image
from typing import Sequence
from tensordict.utils import NestedKey
from tensordict.tensordict import TensorDictBase
from torchrl.data.tensor_specs import TensorSpec, ContinuousBox
from copy import copy
import random
from PIL import Image
from torch.nn import functional as F
import pickle


class LayerNormReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.ln(x)
        x = self.relu(x)
        return x

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

    env = TransformedEnv(
        make_base_env(cfg, device, written_feedback_channel=written_feedback_channel),
        Compose(
            ToTensorImage(
                in_keys=[("agents", "observation", "obs_0_0")], unsqueeze=True
            ),

            CenterCrop(cfg.crop_h, cfg.crop_w, in_keys=[("agents", "observation", "obs_0_0")]),
            Resize(100, 100, in_keys=[("agents", "observation", "obs_0_0")]),
            UnsqueezeTransform(unsqueeze_dim=-3, in_keys=[("agents", "observation", "obs_0_1")]),
        )
    )
    
    if cfg.pretrained_encoder:
        env.append_transform(
            EncoderTransform(
                env_name=cfg.name,
                num_channels=cfg.num_stacks * cfg.num_channels,
                in_keys=[("agents", "observation", "obs_0_0")],
                out_keys=[(("agents", "observation", "encoded_vec"))],
                version="latest",
            )
        )
    env.append_transform(CatFrames(N=cfg.num_stacks, dim=-3, in_keys=[("agents", "observation", "obs_0_0")]))
    env.append_transform(CatFrames(N=cfg.num_stacks, dim=-2, in_keys=[("agents", "observation", "obs_0_1")]))
    env.append_transform(StepCounter())   

    # RandRotate(5, in_keys=[("agents", "observation", "obs_0_0")]),
    # FrameSkipTransform(5),
    # AddCoordinates(in_keys=[("agents", "observation", "obs_0_0")]),
    # RandomShift(0.05, 0.05, in_keys=[("agents", "observation", "obs_0_0")]), # applied in policy

    return env

def make_agent(proof_env, cfg, device):
    action_spec = proof_env.action_spec
    if isinstance(action_spec.space, ContinuousBox):
        model, actor, fb_model = make_agent_continuous(proof_env, cfg, device)
    else:
        model, actor, fb_model = make_agent_discrete(proof_env, cfg, device)

    return model, actor, fb_model

def make_agent_discrete(proof_env, cfg, device):
    # Define Encoder Network
    num_actions = proof_env.action_spec.space.n
    if cfg.envs.pretrained_encoder:
        in_keys = [("agents", "observation", "encoded_vec")]
        encoder = nn.Identity()
        in_dims = 128
        print('Using Pretrained Encoder: ', cfg.envs.name)
    elif cfg.from_states:
        in_keys = [("agents", "observation", "obs_0_1")]
        print('Using States')
        encoder = StateEncoder(cfg.envs.start_dim, cfg.envs.end_dim)
        in_dims = (cfg.envs.end_dim - cfg.envs.start_dim) * cfg.envs.num_stacks
    else:
        in_keys = [("agents", "observation", "obs_0_0")]
        encoder = Encoder(cfg.envs.num_channels, 64)
        in_dims = 64 * cfg.envs.num_stacks

    # Define Actor Network
    actor_net = ActorNet(
        encoder = encoder,
        # n_agent_inputs=64,#proof_env.observation_spec["agents", "encoder_vec"].shape[-1],
        n_agent_inputs=in_dims,
        num_cells=16,
        n_agent_outputs=num_actions,
        activation_class=nn.ReLU,
    )
    # actor_net.load_state_dict(torch.load('crew_algorithms/sac/vanilla.pth')) # vanilla 4/10, 38/100
    # actor_net.load_state_dict(torch.load('../Data/actors_weighed_ce/ep29_it104.pth')) # weighed ce loss 6/10, 19/100
    # actor_net.load_state_dict(torch.load('../Data/actors_entropy/ep14_it459.pth')) # entropy reg 3/10, 34%

    # loaded_state_dict = torch.load('crew_algorithms/sac/vanilla.pth')
    # encoder_state_dict = {key.replace('encoder.', ''): value for key, value in loaded_state_dict.items() if 'encoder.' in key}
    # actor_net.encoder.load_state_dict(encoder_state_dict, strict=False)
    # actor_net.load_state_dict(loaded_state_dict)

    actor_module = SafeModule(
        actor_net,
        in_keys=in_keys,
        out_keys=[
            ("agents", "logits"),
        ],
    )
    actor = ProbabilisticActor(
        spec=proof_env.action_spec,
        in_keys=[("agents", "logits")],
        out_keys=[("agents", "action")],
        module=actor_module,
        distribution_class=Categorical,
        # distribution_class=lambda logits: Categorical(logits=logits),
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )
    # actor = ImitationLearningWrapper(
    #     actor,
    #     action_key=proof_env.action_key,
    #     il_enabled_key=("agents", "observation", "obs_0_1"),
    #     il_action_key=("agents", "observation", "obs_0_1"),
    # )

    # Define Critic Network
    qvalue_net = QValueNet(
        encoder = encoder,
        # n_agent_inputs=64,#proof_env.observation_spec["agents", "encoder_vec"].shape[-1],
        n_agent_inputs=in_dims,
        num_cells=16,
        n_agent_outputs=num_actions,
        activation_class=nn.ReLU, # LayerNormReLU,
    )

    # qvalue_net.encoder.load_state_dict(encoder_state_dict, strict=False)
    # qvalue_net.load_state_dict(loaded_state_dict, strict=False)
    qvalue = ValueOperator(
        in_keys=in_keys,
        out_keys=[("agents", "action_value")],
        module=qvalue_net,
    )


    model = torch.nn.ModuleList([actor, qvalue]).to(device)
    return model, model[0], None



def make_agent_continuous(proof_env, cfg, device):
    action_dims = proof_env.action_spec.space.low.shape[-1]
    additional_in_keys = {tuple(v): k for k, v in cfg.envs.additional_in_keys.items()}


    if cfg.envs.pretrained_encoder:
        in_keys = {("agents", "observation", "encoded_vec"): 'obs'}
        in_keys.update(additional_in_keys)
        encoder = nn.Identity()
        in_dims = 64  * cfg.envs.num_stacks + (cfg.envs.additional_in_keys != {})
    elif cfg.from_states:
        in_keys = {("agents", "observation", "obs_0_1"): 'obs'}
        in_keys.update(additional_in_keys)
        encoder = StateEncoder(cfg.envs.state_start_dim, cfg.envs.state_end_dim)
        in_dims = (cfg.envs.state_end_dim - cfg.envs.state_start_dim) * cfg.envs.num_stacks + (cfg.envs.additional_in_keys != {})
    else:
        in_keys = {("agents", "observation", "obs_0_0"): 'obs'}
        in_keys.update(additional_in_keys)
        encoder = Encoder(cfg.envs.num_channels, 64)
        in_dims = 64 * cfg.envs.num_stacks+ (cfg.envs.additional_in_keys != {})

    actor_net = ContinuousActorNet(
        encoder = encoder,
        n_agent_inputs=in_dims,
        num_cells=256,
        out_dims=action_dims * 2,
    )


    dist_class = TanhNormal
    dist_kwargs = {
        "min": cfg.envs.action_low,
        "max": cfg.envs.action_high,
        "tanh_loc": False,
    }

    actor_extractor = NormalParamExtractor(
        scale_mapping="biased_softplus_1.0",
        scale_lb=1e-4,
        
    )

    actor_net = TensorDictModule(actor_net, in_keys=in_keys, out_keys=["actor_out"])
    actor_extractor = TensorDictModule(actor_extractor, in_keys=["actor_out"], out_keys=["loc", "scale"])

    actor_module = TensorDictSequential(actor_net, actor_extractor)

    actor = ProbabilisticActor(
        spec=proof_env.action_spec,
        in_keys=["loc", "scale"],
        # out_keys=[proof_env.action_key],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )

    # actor = ImitationLearningWrapper(
    #     actor,
    #     action_key=["action"],
    #     il_enabled_key=("agents", "observation", "obs_0_1"),
    #     il_action_key=("agents", "observation", "obs_0_1"),
    # )

    # Define Critic Network
    qvalue_net = ContinuousQValueNet(
        encoder = encoder,
        n_agent_inputs=in_dims + action_dims,
        num_cells=256,
    )

    qvalue = ValueOperator(
        in_keys= {**in_keys, **{("agents", "action"): "action"}},
        module=qvalue_net,
        out_keys=["state_action_value"],
    )

    model = nn.ModuleList([actor, qvalue]).to(device)

    if cfg.feedback_model:
        if cfg.history:
            feedback_in_dims = (in_dims//3) * 7 + 6 * 2 + 6 * 1
        else:
            feedback_in_dims = 64 * cfg.envs.num_stacks + action_dims # only o & a matters
        print('Feedback in dims:', feedback_in_dims)
        feedback_model = ContinuousQValueNet(
            encoder = encoder,
            n_agent_inputs=feedback_in_dims,
            num_cells=256,
        )
        # feedback_model = ValueOperator(
        #     in_keys= {**in_keys, **{("action"): "action"}},
        #     module=feedback_model,
        #     out_keys=["pred_feedback"],
        # )
    else:
        feedback_model = None

    feedback_model = feedback_model.to(device) if feedback_model else None

    return model, model[0], feedback_model
 

def make_data_buffer(cfg, run_name):

    # replay_buffer = TensorDictPrioritizedReplayBuffer(
    #     alpha=0.7,
    #     beta=0.5,
    #     pin_memory=False,
    #     storage=LazyMemmapStorage(cfg.buffer_size, scratch_dir='../Data/Buffer/prb_%s' % run_name),
    #     batch_size=cfg.batch_size,
    #     priority_key="rank_adjusted_td_error",
    # )

    p_sampler = PrioritizedSampler(
        max_capacity=cfg.buffer_size,
        alpha=0.7, 
        beta=0.9, 
        reduction='max')

    replay_buffer = TensorDictReplayBuffer(
        pin_memory=False,
        storage=LazyMemmapStorage(cfg.buffer_size, scratch_dir='../Data/Buffer/prb_%s' % run_name, device='cpu'),
        batch_size=cfg.batch_size,
        # sampler=RandomSampler()
        sampler=p_sampler,
        priority_key=('agents', 'priority_weight'),
    ) 

    if cfg.use_expert:
        replay_buffer_expert = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.5,
            pin_memory=False,
            storage=LazyMemmapStorage(cfg.buffer_size, scratch_dir='../Data/Buffer/prb_%s' % run_name),
            batch_size=cfg.batch_size,
            priority_key="rank_adjusted_td_error",
        )
        replay_buffer_expert = load_prb(replay_buffer_expert)
    else:
        replay_buffer_expert = None


    return replay_buffer, replay_buffer_expert


def make_loss_module(cfg, env, model):
    """Make loss module and target network updater."""
    # Create SAC loss
    if isinstance(env.action_spec.space, ContinuousBox):
        print('Continuous SAC')
        loss_module = SACLoss(
            actor_network=model[0],
            action_spec=env.action_spec,
            qvalue_network=model[1],
            alpha_init=cfg.optimization.alpha_init,
            target_entropy=cfg.optimization.target_entropy,
            num_qvalue_nets=2,
            delay_actor=False,
            delay_qvalue=True,        
            loss_function="l2",
        )
        loss_module.set_keys(
            action=env.action_key,
            reward=env.reward_key,
            done=("agents", "done"),
            priority=("agents", "td_error"),
        )

    else:
        print('Discrete SAC')
        num_actions = env.action_spec.space.n
        loss_module = DiscreteSACLoss(
            actor_network=model[0],
            action_space=env.action_spec,
            qvalue_network=model[1],
            # alpha_init=cfg.optimization.alpha_init,
            num_actions=num_actions,
            num_qvalue_nets=2,
            target_entropy_weight=cfg.optimization.target_entropy_weight,
            # target_entropy="auto",
            # target_entropy=cfg.optimization.target_entropy,
            # fixed_alpha=0.05,
            loss_function="l2",
        )
        loss_module.set_keys(
            action=env.action_key,
            reward=env.reward_key,
            done=("agents", "done"),
            # done=env.done_key,
            value=("agents", "state_value"), 
            action_value=("agents", "action_value"),
            priority=("agents", "td_error"),            
        )

    loss_module.make_value_estimator(gamma=cfg.optimization.gamma)
    target_net_updater = SoftUpdate(loss_module, eps=cfg.optimization.target_update_polyak)
    return loss_module, target_net_updater

def make_optimizer(cfg, loss_module):
    # Define Target Network Updater
    # target_net_updater = SoftUpdate(
    #     loss_module, eps=cfg.optimization.target_update_polyak
    # )
    # critic_params = list(loss_module.qvalue_network_params['module', 'mlp'].flatten_keys().values())
    # actor_params = list(loss_module.actor_network_params['module', '0', 'module', '0', 'mlp'].flatten_keys().values())
    # params = critic_params + actor_params

    # Define optimizer
    optimizer = optim.Adam(
        loss_module.parameters(),
        # params,
        lr=cfg.optimization.lr,
        weight_decay=cfg.optimization.weight_decay,
    )
    
    return optimizer


def make_sac_optimizer(cfg, loss_module):
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())
    # critic_params = list(loss_module.qvalue_network_params['module', 'mlp'].flatten_keys().values())
    # actor_params = list(loss_module.actor_network_params['module', '0', 'module', 'mlp'].flatten_keys().values())# + list(loss_module.qvalue_network_params['module', 'encoder'].flatten_keys().values())

    optimizer_actor = optim.Adam(
        actor_params,
        lr=cfg.optimization.lr,
        # eps=1e-4,
        weight_decay=cfg.optimization.weight_decay,
    )
    optimizer_critic = optim.Adam(
        critic_params,
        lr=cfg.optimization.lr,
        # eps=1e-4,
        weight_decay=cfg.optimization.weight_decay,
    )
    optimizer_alpha = optim.Adam(
        [loss_module.log_alpha],
        # eps=1e-4,
        lr=3.0e-4,
    )

    return optimizer_actor, optimizer_critic, optimizer_alpha


def lr_scheduler(optimizer_actor, optimizer_critic, step, phase_1=40, phase_2=60, lr_1=5e-5, lr_2=1.5e-5):
    if step < phase_1:
        optimizer_actor.param_groups[0]["lr"] = 0
        optimizer_critic.param_groups[0]["lr"] = lr_1
        phase = 'critic learning'

    elif step < phase_2:
        optimizer_actor.param_groups[0]["lr"] = (step - phase_1) * lr_2 / (phase_2 - phase_1)
        optimizer_critic.param_groups[0]["lr"] = lr_1 + (step - phase_1) * (lr_2 - lr_1) / (phase_2 - phase_1)
        phase = 'warmup'

    else:
        optimizer_actor.param_groups[0]["lr"] = lr_2
        optimizer_critic.param_groups[0]["lr"] = lr_2
        phase = 'interactive learning'

    return optimizer_actor, optimizer_critic, phase
    
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
    human_feedback = data[..., -1, 0]

    # human_feedback_0 = (
    #     torch.ones((N, *human_feedback.shape[1:]), device=human_feedback.device) * -9
    # )

    human_feedback_0 = (
        torch.ones((N, *human_feedback.shape[1:]), device=human_feedback.device) * 0
    )

    human_feedback = torch.cat([human_feedback, human_feedback_0], 0)
    human_feedback = human_feedback[N:]

    data[..., -1, 0] = human_feedback
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
    feedback = original_feedback[..., -1, 0]

    # print(td.get(il_enabled_key))
    il_enabled = td.get(il_enabled_key)[..., -1, 4].bool()
    # print('IL on:', il_enabled.sum())
    feedback[il_enabled] = il_feedback
    original_feedback[..., -1, 0] = feedback
    td.set(feedback_key, original_feedback)
    return td


def audio_feedback(stream, time_stamp, action_key, reward_key, prb):
    # print(len(stream.buffer))
    if len(stream.buffer) < 128:
        return prb

    # TODO: can also try not saving to a file

    effected_data_idx = prb["_data", "time_stamp"] > time_stamp - 3

    file_name = "audio_%.2f" % time_stamp
    stream.save_to_file(file_name)
    reward_dict = audio2reward(
        "crew_algorithms/sac/audio/%s.wav" % file_name
    )
    # audio_tensor = stream.to_torch_tensor()
    # reward_dict = audio2reward(audio_tensor)


    audio_reward = torch.tensor([reward_dict[a.item()] for a in \
                                 prb["_data", action_key][effected_data_idx]\
                                    .numpy()]).unsqueeze(-1).unsqueeze(-1)

    prb["_data", reward_key][effected_data_idx] += audio_reward

    # print(prb[effected_data_idx][reward_key].sum())
    return prb


def combine_feedback_and_rewards(td,feedback_key, reward_key, scale_feedback=1.0):
    # print('Feedback Received = %.4f' % (td.get(feedback_key).sum().item()))
    # if len(td.get(feedback_key)) >0:
    #     print('Feedback Received = %.4f' % (td.get(feedback_key).item()))
    td[reward_key] = td[reward_key]  + td.get(feedback_key) * scale_feedback
    return td


def get_time():
    now = datetime.now()
    return now.strftime("%m%d_%H%M")


def fill_prb(demonstration_folder):
    """
    Data dict Format:

TensorDict(
    fields={
        agents: TensorDict(
            fields={
                action: Tensor(shape=torch.Size([256, 1]), device=cuda:0, dtype=torch.int64, is_shared=True),
                done: Tensor(shape=torch.Size([256, 1, 1]), device=cuda:0, dtype=torch.bool, is_shared=True),
                logits: Tensor(shape=torch.Size([256, 1, 4]), device=cuda:0, dtype=torch.float32, is_shared=True),
                observation: TensorDict(
                    fields={
                        obs_0_0: Tensor(shape=torch.Size([256, 1, 3, 128, 128]), device=cuda:0, dtype=torch.float32, is_shared=True),
                        obs_0_1: Tensor(shape=torch.Size([256, 1, 13]), device=cuda:0, dtype=torch.float32, is_shared=True)},
                    batch_size=torch.Size([256, 1]),
                    device=cuda:0,
                    is_shared=True)},
            batch_size=torch.Size([256, 1]),
            device=cuda:0,
            is_shared=True),
        collector: TensorDict(
            fields={
                traj_ids: Tensor(shape=torch.Size([256]), device=cuda:0, dtype=torch.int64, is_shared=True)},
            batch_size=torch.Size([256]),
            device=cuda:0,
            is_shared=True),
        is_expert: Tensor(shape=torch.Size([256]), device=cuda:0, dtype=torch.bool, is_shared=True),
        next: TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([256, 1, 1]), device=cuda:0, dtype=torch.bool, is_shared=True),
                        observation: TensorDict(
                            fields={
                                obs_0_0: Tensor(shape=torch.Size([256, 1, 3, 128, 128]), device=cuda:0, dtype=torch.float32, is_shared=True),
                                obs_0_1: Tensor(shape=torch.Size([256, 1, 13]), device=cuda:0, dtype=torch.float32, is_shared=True)},
                            batch_size=torch.Size([256, 1]),
                            device=cuda:0,
                            is_shared=True),
                        reward: Tensor(shape=torch.Size([256, 1, 1]), device=cuda:0, dtype=torch.float32, is_shared=True)},
                    batch_size=torch.Size([256, 1]),
                    device=cuda:0,
                    is_shared=True),
                step_count: Tensor(shape=torch.Size([256, 1, 1]), device=cuda:0, dtype=torch.int64, is_shared=True)},
            batch_size=torch.Size([256]),
            device=cuda:0,
            is_shared=True),
        step_count: Tensor(shape=torch.Size([256, 1, 1]), device=cuda:0, dtype=torch.int64, is_shared=True),
        time_stamp: Tensor(shape=torch.Size([256]), device=cuda:0, dtype=torch.float32, is_shared=True)},
    batch_size=torch.Size([256]),
    device=cuda:0,
    is_shared=True)

    """


def fill_prb(act_folder='../Data/act4', obs_folder='../Data/obs4', ending_steps_path='../Data/ending_steps.pth'):
    import re
    import os
    from PIL import Image
    from torchvision import transforms
    from tensordict import TensorDict

    to_tensor = transforms.ToTensor()

    def sorting_key(string):
        numbers = re.findall(r'\d+', string)
        return int(numbers[0]), int(numbers[1])

    o_list = [f for f in os.listdir(obs_folder)]
    o_list = sorted(o_list, key=sorting_key)

    a_list = [f for f in os.listdir(act_folder)]
    a_list = sorted(a_list, key=sorting_key)

    ending_steps = torch.load(ending_steps_path)
    traj_ID = 0

    experiences = []

    for i in range(len(o_list)-1):
        print('Processing %d/%d' % (i, len(o_list)-1), end='\r')

        o_i = o_list[i][3:-4]
        a_i = a_list[i][3:-3]
        assert o_i == a_i, 'Not match: %s, %s' % (o_i, a_i)

        obs_i = to_tensor(Image.open(os.path.join(obs_folder, o_list[i])).convert('RGB')).unsqueeze(0)
        act_i = torch.tensor(torch.load(os.path.join(act_folder, a_list[i])), dtype=torch.int64).unsqueeze(0)
        logits_i = torch.zeros((1, 4))
        logits_i[0, int(act_i.item())] = 1

        next_obs_i = to_tensor(Image.open(os.path.join(obs_folder, o_list[i+1])).convert('RGB')).unsqueeze(0)
        next_act_i = torch.tensor(torch.load(os.path.join(act_folder, a_list[i+1])), dtype=torch.int64).unsqueeze(0)
        next_logits_i = torch.zeros((1, 4))
        next_logits_i[0, int(next_act_i.item())] = 1

        reward = torch.tensor([[[0.000]]])
        done = torch.tensor([[[False]]])

        if i in ending_steps:
            reward = torch.tensor([[[1.0]]])
            # done = torch.tensor([[[True]]])
        if i-1 in ending_steps:
            continue
        if i-2 in ending_steps:
            traj_ID += 1

        

        data_dict = TensorDict({
            'agents': TensorDict({
                'action': act_i.unsqueeze(0),
                'done': torch.tensor([[[False]]]),
                'logits': logits_i.unsqueeze(1),
                'observation': TensorDict({
                    'obs_0_0': obs_i.unsqueeze(0),
                    'obs_0_1': torch.zeros((1, 1, 13)),
                },
                batch_size=[1, 1], device='cuda')
            }, batch_size=[1, 1], device='cuda'),
            'collector': TensorDict({'traj_ids': torch.tensor([traj_ID])},
                                    batch_size=[1], device='cuda'),
            'next': TensorDict({
                'agents': TensorDict({
                    'done': done,
                    'observation': TensorDict({
                        'obs_0_0': next_obs_i.unsqueeze(0),
                        'obs_0_1': torch.zeros((1, 1, 13)),
                    }, batch_size=[1, 1], device='cuda'),
                    'reward': reward
                }, batch_size=[1, 1], device='cuda'),
                'step_count': torch.zeros((1, 1, 1), dtype=torch.int64),
            }, batch_size=[], device='cuda'),
            'step_count': torch.zeros((1, 1, 1), dtype=torch.int64),
            'time_stamp': - torch.ones((1)) * 100,
            'is_expert': torch.tensor([False]),
        }, batch_size=[1], device='cuda')

        experiences.append(data_dict)

        if (i+1)%10000==0 or i==66810:
            torch.save(experiences, '../Data/Replay_nodone/experiences_%d.pt' % i)
            experiences = []
    

    # print(len(experiences))


        # finding the episode ending steps
        # if torch.abs(next_obs_i - obs_i).sum() > 1000:
        #     print(torch.abs(next_obs_i - obs_i).sum(), i)
        #     num_eps += 1
        #     ending_steps.append(i-1)

    # for f in f_list:
    #     # new_name = f[:1]+ '_0' + f[1:]
    #     os.rename(os.path.join(obs_folder, f), os.path.join(obs_folder, f[:1] + '_0' + f[1:]))



def load_prb(prb, path='../Data/Replay_nodone', chunk_size=1024):
    from time import time
    tic = time()
    exp_list = [f for f in os.listdir(path) if f[-2:] == 'pt']
    for i, f in enumerate(exp_list):
        print('Loading Expert Experiences: %d/%d' % (i, len(exp_list)), end='\r')
        experiences = torch.load(os.path.join(path, f))
        # import pdb; pdb.set_trace()
        for j in range(0, len(experiences), chunk_size):
            # print('Loading Expert Experiences: %d/%d' % (j, len(experiences)), end='\r')
            e = torch.cat(experiences[j:j+chunk_size], dim=0)
            prb.extend(e.cpu())

        # for j, e in enumerate(experiences):
        #     print('Loading Expert Experiences: %d/%d' % (j, len(experiences)), end='\r')
            # experiences['batch_size'] = [1]
            # experiences = torch.cat(experiences, dim=0)
            # print(e)
            # prb.extend(e)
    print('Loading finished in %.1f seconds' % (time() - tic))
    return prb

def visualize(data, i, num_channels, fpb, hf=False, continuous=True):
    for j in range(len(data)):
        r = data["next", "agents", "reward"][j].item()
        d = data["next", "agents", "done"][j].item()

        # a = data["action"][j]
        # x, y = a[..., 0], a[..., 1]
        if hf:
            r_hf = data["agents", "observation", "obs_0_1"][j][..., -1, 0].item()
        else:
            r_hf = data["next", "agents", "feedback"][j].item()

        save_image(data["agents", "observation", "obs_0_0"][j, ..., -num_channels:, :, :], 'visualize/frame_%d_r%.2f_d%d_rhf%.2f.png' % (i*fpb + j, r, d, r_hf))
        save_image(data["next", "agents", "observation", "obs_0_0"][j, ..., -num_channels:, :, :], 'visualize/frame_%d_next.png' % (i*fpb + j))


        # a = ['f', 'b', 'l', 'r'][data["agents", "action"][j].int().item()]
        # save_image(data["agents", "observation", "obs_0_0"][j], 'visualize/frame_%d_%s_r%d_d%d.png' % (i*128 + j, a, r, d))
        # save_image(data["next", "agents", "observation", "obs_0_0"][j], 'visualize/frame_%d_next.png' % (i*128 + j))

        # cx, cy = max(min(int((-x + 10) * 5), 99), 0), max(min(int((y + 10) * 5), 99), 0)

        # current_frame = data["agents", "observation", "obs_0_0"][j][:, -3:]
        # current_frame[:, :, max(cx-2, 0): min(cx+2, 99), max(cy-1, 0): min(cy+1, 99)] = 1
        # current_frame[:, :, max(cx-1, 0): min(cx+1, 99), max(cy-2, 0): min(cy+2, 99)] = 1

        
        # save_image(current_frame, 'visualize/frame_%d_%.1f_%.1f_rhf_%.2f_r%d_d%d.png' % (i*128 + j, x, y,  r_hf, r, d))
        # save_image(data["next", "agents", "observation", "obs_0_0"][j][:, -3:], 'visualize/frame_%d_next.png' % (i*128 + j))


class heuristic_feedback():
    def __init__(self, template_path, threshold, batch_size, device):
        template = Image.open(template_path)
        self.totens = transforms.ToTensor()
        self.template = self.totens(template).unsqueeze(0).to(device) + 1e-6

        torch._assert(len(self.template.shape) == 4, "Template should be 4D")
        self.template_morm = ((self.template ** 2).sum() ** 0.5).repeat(batch_size, 1, 1, 1)
        self.one_kernel = torch.ones_like(self.template, device=device)
        self.threshold = threshold

    def treasure_in_view(self, frame):
        torch._assert(len(frame.shape) == 4, "Frame should be 4D")

        frame = frame + 1e-6

        brightness = F.conv2d(frame ** 2, self.one_kernel) ** 0.5
        heat_map = F.conv2d(frame, self.template)
        heat_map = heat_map / (brightness * self.template_morm)

        out = (heat_map>self.threshold).sum(dim=(1, 2, 3)) > 0 
        return out

    def moved_closer(self, td):
        next_agent = td["next", "agents", "observation", "obs_0_1"][..., -1,  5:8]
        next_treasure = td["next", "agents", "observation", "obs_0_1"][..., -1,  11:]
        next_agent[..., 1], next_treasure[..., 1] = 0, 0
        next_distance = ((next_agent - next_treasure) ** 2).mean(dim=-1, keepdim=True)

        current_agent = td["agents", "observation", "obs_0_1"][..., -1,  5:8]
        current_treasure = td["agents", "observation", "obs_0_1"][..., -1, 11:]
        current_agent[..., 1], current_treasure[..., 1] = 0, 0
        current_distance = ((current_agent - current_treasure) ** 2).mean(dim=-1, keepdim=True)

        r_dis = (current_distance - next_distance) / 10
        # r_dis = (torch.sign(current_distance - next_distance))
        return r_dis.squeeze(-1).squeeze(-1)

    def explored(self, f_current, f_next):
        non_black_current = (f_current > 0).float().sum(dim=(1, 2, 3))
        non_black_next = (f_next > 0).float().sum(dim=(1, 2, 3))
        explored = (non_black_next - non_black_current - 300)/1000
        # explored = (non_black_next > (non_black_current + 300)).float() * 2 - 1
        return explored

    def treasure_appeared(self, f_current, f_next):
        current_treasure = self.treasure_in_view(f_current)
        next_treasure = self.treasure_in_view(f_next)
        return (next_treasure & ~current_treasure).float() * 2 - 1

    def get_treasure_in_view(self, f_current, f_next):
        in_current = self.treasure_in_view(f_current)
        in_next = self.treasure_in_view(f_next)
        return in_current, in_next

    def provide_feedback(self, td):
        
        f_current = td.get(("agents", "observation", "obs_0_0")).squeeze(1)[:, -3:]
        f_next = td.get(("next", "agents", "observation", "obs_0_0")).squeeze(1)[:, -3:]
    
        treasure_in_view, treasure_in_next = self.get_treasure_in_view(f_current, f_next)
        treasure_not_in_view = ~treasure_in_view

        moved_closer = self.moved_closer(td)
        explored = self.explored(f_current, f_next)

        feedback = treasure_in_view.float() * moved_closer  + treasure_not_in_view.float() * explored + (treasure_in_next.float() - treasure_in_view.float()) * 3

        # feedback = torch.rand_like(feedback) * 2 - 1 # random feedback
        feedback = feedback 
 
        return feedback.unsqueeze(-1).unsqueeze(-1) 


def save_training(model, prb, episode_success, loss_module, run_name, i):
    os.makedirs("../Data/Pretrain/%s" % run_name, exist_ok=True)
    with open('../Data/Pretrain/%s/sr.pkl' % run_name, 'wb') as f:
        pickle.dump(episode_success, f)
    torch.save(model.state_dict(), "../Data/Pretrain/%s/weights_Iter_%d.pth" % (run_name, i))
    torch.save(loss_module.state_dict(), "../Data/Pretrain/%s/loss_module_Iter_%d.pth" % (run_name, i))
    prb.dumps('../Data/Pretrain/%s/prb.pkl' % run_name)


def load_training(model, prb, loss_module, run_name, global_start_time, i):
    model.load_state_dict(torch.load("../Data/Pretrain/%s/weights_Iter_%d.pth" % (run_name, i)))
    loss_module.load_state_dict(torch.load("../Data/Pretrain/%s/loss_module_Iter_%d.pth" % (run_name, i)))
    prb.loads('../Data/Pretrain/%s/prb.pkl' % run_name)
    with open('../Data/Pretrain/%s/sr.pkl' % run_name, 'rb') as f:
        episode_success = pickle.load(f)
    collected_frames = i * 128
    time_stamp = collected_frames / 2
    # prb.update_priority(index=torch.arange(len(prb)), priority=torch.ones(len(prb)) * 0.25)
    return model, model[0], prb, episode_success, loss_module, time_stamp + global_start_time, collected_frames


def feedback_model_train_step(history, model, data, optim, val=False):

    if history:
        obs = data["agents", "history", "obs"] # [bs, 1, 3x7, 100, 100]
        action = data["agents", "history", "actions"] # [bs, 6, 2]
        feedback = data["agents", "history", "feedbacks"] # [bs, 6, 1]
        
    else:
        obs = data["next", "agents", "observation", "obs_0_0"] # a stack of t-1, t, t+1
        action = data["agents", "action"]
        feedback = data["next", "agents", "feedback"]

    # predicted_feedback = model(data)["pred_feedback"]
    predicted_feedback = model(obs=obs, action=action)

    if val:
        return (predicted_feedback - feedback).pow(2).mean().item()

    optim.zero_grad()
    loss = (predicted_feedback - feedback).pow(2).mean()
    loss.backward()
    optim.step()

    return loss.item()
    
def provide_learned_feedback(history, model, data):
    if history:
        obs = data["agents", "history", "obs"] # [bs, 1, 3x7, 100, 100]
        action = data["agents", "history", "actions"] # [bs, 6, 2]
        feedback = data["agents", "history", "feedbacks"] # [bs, 6, 1]
        
    else:
        obs = data["next", "agents", "observation", "obs_0_0"] # a stack of t-1, t, t+1
        action = data["agents", "action"]
        feedback = data["next", "agents", "feedback"]

    with torch.inference_mode():
        predicted_feedback = model(obs=obs, action=action)
    return predicted_feedback



if __name__ == '__main__':
    fill_prb()