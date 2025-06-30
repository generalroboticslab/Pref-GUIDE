# log smoothing
# done signal
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
from torch.optim import Adam

from crew_algorithms.envs.configs import EnvironmentConfig, register_env_configs
from crew_algorithms.sac.audio_feedback import Audio_Streamer
from crew_algorithms.sac.config import NetworkConfig, OptimizationConfig
from crew_algorithms.sac.logger import custom_logger
from crew_algorithms.utils.wandb_utils import WandbConfig
from tensordict import TensorDict

@define(auto_attribs=True)
class Config:
    envs: EnvironmentConfig = MISSING
    """Settings for the environment to use."""
    optimization: OptimizationConfig = OptimizationConfig()
    network: NetworkConfig = NetworkConfig()
    collector: OffPolicyCollectorConfig = OffPolicyCollectorConfig(
        frames_per_batch=240, init_random_frames=10
    )
    """Settings to use for the off-policy collector."""
    wandb: WandbConfig = WandbConfig(entity="lingyu-zhang", project="crew-final-continue")
    """WandB logger configuration."""
    batch_size: int = 240
    buffer_size: int = 5_000
    num_envs: int = 1
    seed: int = 42
    from_states: bool = False
    audio: bool = False
    traj: bool = False
    use_expert: bool = False
    heuristic_feedback: bool = False
    hf: bool = False    
    feedback_model: bool = False
    deploy_feedback_model_ratio: int = 2
    history: bool = False
    log_smoothing: int = 100
    continue_training: str = 'none'
    sub_name: str = 'none'

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
register_env_configs()

@hydra.main(version_base=None, config_path="../conf", config_name="sac")
def sac(cfg: Config):

    import os
    import uuid
    from collections import deque, defaultdict

    import torch
    import random
    import numpy as np
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
        lr_scheduler,
        get_time,
        visualize,
        heuristic_feedback,
        save_training,
        load_training,
        feedback_model_train_step,
        provide_learned_feedback
    )
    from crew_algorithms.utils.rl_utils import make_collector

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    cfg.envs.seed = cfg.seed

    wandb.login()

    exp = 'sachf' if cfg.hf else 'sac'
    is_heu = 'heuristic' if cfg.heuristic_feedback else ''
    run_name = get_time() + '_' + cfg.envs.name  + '_' + is_heu + '_' + exp + '_seed_' + str(cfg.seed) 

    logger = get_logger(
        "wandb",
        logger_name=os.getcwd(),
        experiment_name=run_name,
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

    model, actor, feedback_model = make_agent(env, cfg, device)
    loss_module, target_net_updater = make_loss_module(cfg, env, model)
    env.close()

    prb, prb_e = make_data_buffer(cfg, run_name)
    collected_frames = 0
    episode_success = []
    global_start_time = time()

    if cfg.continue_training != 'none':
        model, actor, prb, episode_success, loss_module, global_start_time, collected_frames = load_training(model, prb, loss_module, cfg.continue_training,  global_start_time, 5, device)
        stop_iter = 10
        feedback_model.encoder.load_state_dict(model[1].encoder.state_dict())
    else:
        if cfg.hf:
            stop_iter = 5
        else:
            stop_iter = 15
        
    local_logger = custom_logger(path='crew_algorithms/sac/logs/' + run_name + '.json', start_time=global_start_time)

    collector = make_collector(cfg.collector, env_fn, actor, device, cfg.num_envs)
    collector.set_seed(cfg.seed)
    
    human_delay_buffer_td = None

    # if cfg.audio:
    #     stream = Audio_Streamer()
    #     stream.start_streaming()

    episode_reward, episode_reward_hf = defaultdict(float), defaultdict(float)

    num_success, num_trajs, last_traj = 0, 0, 0
    loss = None

    heuristic = heuristic_feedback(cfg.envs.target_img, 0.95, cfg.collector.frames_per_batch, device)
    optimizer = make_optimizer(cfg, loss_module)

    if cfg.feedback_model:
        feedback_optimizer = Adam(feedback_model.parameters(), lr=1e-4)

    deploy_learned_feedback = False

    all_hf, all_heu = [], []

    stage = '_continued' if cfg.continue_training != 'none' else ''
    os.makedirs(f"../Data/FINAL/{cfg.sub_name}/{exp}/{run_name}{stage}", exist_ok=True)
    torch.save(model.state_dict(), f"../Data/FINAL/{cfg.sub_name}/{exp}/{run_name}{stage}/0.pth")


    for i, data in enumerate(collector):
        time_stamp = time() - global_start_time
            
        logger.log_scalar("time", time_stamp, step=collected_frames)
        local_logger.log(x_axis="steps", y_axis="learned_feedback_deployed", x_value=collected_frames, y_value=deploy_learned_feedback, log_time=True)

        collector.update_policy_weights_()
        data = data.view(-1)
        
        num_trajs += data["next", "agents", "done"].sum().item() # Here the reward is fresh from unity, being pos=1, neg=0. So this is: if failed, set done to false

        episode_success.extend(data["next", "agents", "reward"][data["next", "agents", "done"]==True].tolist()) # all the done ones are either success or terminated failure, so 1 is success, 0 is failure
        # data["next", "agents", "done"][data["next", "agents", "reward"]==0] = False # Set to Done for Finite Horizon 
        num_success += int(data["next", "agents", "reward"].sum().item())

        data["done"] = data["agents", "done"]
        
        current_frames = data.numel()
        collected_frames += current_frames

        if cfg.envs.name in ['find_treasure', 'hide_and_seek_1v1']:
            data.set(('next', 'agents', 'heuristic_feedback'), heuristic.provide_feedback(data))

        if cfg.heuristic_feedback: 
            data.set(('next', 'agents', 'feedback'), data[('next', 'agents', 'heuristic_feedback')])
        else:
            data.set(('next', 'agents', 'feedback'), torch.zeros_like(data[('next', 'agents', 'reward')]).to(device))

        # if i < 4 and not cfg.hf:
        #     visualize(data, i, cfg.envs.num_channels, cfg.collector.frames_per_batch, hf=False)

        for j in range(len(data)):
            time_stamp = data[("next", "agents", "observation", "obs_0_1")][j, ..., -1, 1].item()
                
            traj_j = data["agents", "observation", "obs_0_1"][j, ..., -1, 2].int().item()
            episode_reward[traj_j] += data["next", "agents", "reward"][j].item()


            if cfg.hf: 
                episode_reward_hf[traj_j] += data["agents", "observation", "obs_0_1"][j, ..., -1, 0].item() 
            
            if traj_j> last_traj:
  
                r_list = [episode_reward[ep] for ep in episode_reward]
                avg_ep_r = np.array(r_list[max(0, len(r_list)-cfg.log_smoothing - 1):-1]).mean()
                logger.log_scalar("avg_episode_reward", avg_ep_r, step=collected_frames)
                local_logger.log(x_axis="steps", y_axis="avg_episode_reward", x_value=collected_frames, y_value=avg_ep_r, log_time=True)


                if cfg.hf:
                    r_list_hf = [episode_reward_hf[ep] for ep in episode_reward_hf]
                    avg_ep_r_hf = np.array(r_list_hf[max(0, len(r_list_hf)-cfg.log_smoothing - 1):-1]).mean()
                    logger.log_scalar("avg_episode_reward_hf", avg_ep_r_hf, step=collected_frames)
                    local_logger.log(x_axis="steps", y_axis="avg_episode_reward_hf", x_value=collected_frames, y_value=avg_ep_r_hf, log_time=True)
    

                sr_list = [1 if episode_reward[ep] > 0 else 0 for ep in episode_reward]
                avg_sr = np.array(sr_list[max(0, len(sr_list)-cfg.log_smoothing - 1):-1]).mean()
                logger.log_scalar("success_rate", avg_sr, step=collected_frames)
                local_logger.log(x_axis="steps", y_axis="success_rate", x_value=collected_frames, y_value=avg_sr, log_time=True)

                last_traj = traj_j

        # log_file[1] = {k: round(v, 4) for k,v in episode_reward.items()}
        local_logger.data["all_rewards"] = {k: round(v, 4) for k,v in episode_reward.items()}

        # augment environment reward
        data.set(('next', 'agents', 'reward'), data[('next', 'agents', 'reward')] * cfg.envs.scale_reward + cfg.envs.shift_reward) # set reward to defined pos & neg

        data["time_stamp"] = torch.tensor([round(time_stamp, 4)] * data.numel())
        data.set(('is_expert'), torch.tensor([False] * data.numel()))

        data["agents", "observation", "obs_0_1"][data["agents", "observation", "obs_0_1"]==-9] = 0
        data["next", "agents", "observation", "obs_0_1"][data["next", "agents", "observation", "obs_0_1"]==-9] = 0
 
        
        if cfg.hf:
            # save hf values and heurisitc values
            if cfg.envs.name in ['find_treasure', 'hide_and_seek_1v1']:
                hf_values = data["agents", "observation", "obs_0_1"][..., -1, 0]
                all_hf.extend(hf_values.squeeze(1).tolist())
                all_heu.extend(data[('next', 'agents', 'heuristic_feedback')].squeeze(1).squeeze(1).tolist())


            if human_delay_buffer_td is not None:
                data = torch.cat([human_delay_buffer_td, data], dim=0)

            human_delay_td, human_delay_buffer_td = human_delay_transform(
                data, ("agents", "observation", "obs_0_1"), cfg.envs.human_delay_steps
            )

            # grad_average_td = gradient_weighted_average_transform(
            #     human_delay_td, ("agents", "observation", "obs_0_1"), 5
            # )
            grad_average_td = human_delay_td
            il_feedback_td = grad_average_td
            # il_feedback_td = override_il_feedback(
            #     grad_average_td,
            #     ("agents", "observation", "obs_0_1"),
            #     ("agents", "observation", "obs_0_1"),
            #     1,
            # )

            il_feedback_td.set(('next', 'agents', 'feedback'), il_feedback_td.get(("agents", "observation", "obs_0_1"))[..., -1, 0].unsqueeze(dim=-1)) # write human feedback to feedback field
        else:
            il_feedback_td = data

        combined_rewards = combine_feedback_and_rewards(
            il_feedback_td,
            # ("next", "agents", "observation", "obs_0_1"),
            ("next", "agents", "feedback"),
            ("next", "agents", "reward"),
            cfg.envs.dense_reward_scale,
        ) # add feedback to reward



        if cfg.history:
        
            if i==0:
                last_data = combined_rewards[:6]


            new_combined_rewards = torch.cat([last_data, combined_rewards], dim=0)

            obs, actions, feedbacks = [], [], []
   
            for _ in range(len(combined_rewards)):
                # past 6 frames + 1 future frame
                o = new_combined_rewards.get(("agents", "observation", "obs_0_0"))[_: _+7, 0, -3:, ...].reshape(-1, 100, 100)
                a = new_combined_rewards.get(("agents", "action"))[_: _+6, 0].reshape(-1, 2)
                f = new_combined_rewards.get(("next", "agents", "feedback"))[_: _+6, 0].reshape(-1, 1)
                obs.append(o)
                actions.append(a)
                feedbacks.append(f)

            # no need for the following because ("next", "agents", "feedback") != 0 examples will be filtered out later
                
            # if deploy_learned_feedback is False: # only use for training if the learned feedback is not deployed
            #     use_for_train = torch.ones_like(combined_rewards.get(("next", "agents", "reward"))).to(device)
            # else:
            #     use_for_train = torch.zeros_like(combined_rewards.get(("next", "agents", "reward"))).to(device)

            obs = torch.stack(obs, dim=0)
            actions = torch.stack(actions, dim=0)
            feedbacks = torch.stack(feedbacks, dim=0)

            history = TensorDict(
                {
                    "obs": obs.unsqueeze(1),
                    "actions": actions.unsqueeze(1),
                    "feedbacks": feedbacks.unsqueeze(1),
                    # "use_for_train": use_for_train.unsqueeze(1)
                },
                batch_size=combined_rewards.batch_size,
            )
            combined_rewards.set(("agents", "history"), history)


        if deploy_learned_feedback and cfg.feedback_model:
            print('Providing Learned Feedback:', provide_learned_feedback(cfg.history, feedback_model, combined_rewards).sum())
            combined_rewards.set(("next", "agents", "reward"), combined_rewards.get(("next", "agents", "reward")) + provide_learned_feedback(cfg.history, feedback_model, combined_rewards) * cfg.envs.dense_reward_scale)

        print('Rewards:', combined_rewards.get(("next", "agents", "reward")).sum().item())
        if len(combined_rewards) > 0:
            prb.extend(combined_rewards.cpu())

        # if cfg.audio:
        #     prb = audio_feedback(
        #         stream,
        #         time_stamp,
        #         ("agents", "action"),
        #         ("next", "agents", "reward"),
        #         prb,
        #     )

        total_collected_epochs = (
            data[("agents", "observation", "obs_0_1")][..., -1, 2].int().max().item()
        )
        with open('written_feedback_queue.pkl', 'rb') as f:
            written_feedback_queue = pickle.load(f)
            while total_collected_epochs > len(ranked_trajectories) and len(written_feedback_queue) > 0 and cfg.traj:
                trajectory_feedback = TrajectoryFeedback.from_id_and_str(
                    len(ranked_trajectories), written_feedback_queue.popleft().message
                )
                ranked_trajectories.add(trajectory_feedback)
        with open('written_feedback_queue.pkl', 'wb') as f:
            pickle.dump(written_feedback_queue, f)

        print("\n------- Iter: %d | Traj %d| Time: %.2f -------" %(i, num_trajs, time() - global_start_time))
        print('num_success: %d' % (num_success))

        t1, t2 = [], []
        if collected_frames >= cfg.collector.init_random_frames:
            (
                total_losses,
                actor_losses,
                q_losses,
                alpha_losses,
                alphas,
                entropies,
                feedback_model_losses,
            ) = ([], [], [], [], [], [], [])

            for _ in range(
                int(cfg.collector.frames_per_batch * (cfg.optimization.utd_ratio))
            ):
                tic = time()

                # if cfg.audio:
                #     stream.get_sample()

                # sample from replay buffer
                if cfg.use_expert:
                    sampled_expert = prb_e.sample(batch_size=cfg.batch_size//2).clone().to(device)
                    sampled_new = prb.sample(batch_size=cfg.batch_size//2).clone().to(device)
                    sampled_tensordict = torch.cat([sampled_expert, sampled_new], dim=0)
                else:
                    sampled_tensordict = prb.sample().clone().to(device)

                loss_td = loss_module(sampled_tensordict)

                actor_loss = loss_td["loss_actor"]
                q_loss = loss_td["loss_qvalue"]
                alpha_loss = loss_td["loss_alpha"]

                optimizer.zero_grad()
                loss = actor_loss + q_loss + alpha_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimization.max_grad_norm)
                optimizer.step()

                if cfg.traj:
                    traj_ids_to_ranking_map = build_traj_id_to_ranking_map(
                        ranked_trajectories
                    )
                    ranking = (
                        sampled_tensordict[("agents", "observation", "obs_0_1")][..., -1, 2]
                        .int()
                        .cpu()
                        .apply_(traj_ids_to_ranking_map.get)
                        .to(device)
                    )
                    sampled_tensordict["rank_adjusted_td_error"] = (
                        ranking * sampled_tensordict["agents", "td_error"]
                    )

                target_net_updater.step()
                t1.append(time() - tic)
                # if (_ + 1) % 1 == 0:
                #     if cfg.use_expert:
                #         sampled_expert = sampled_tensordict[:cfg.batch_size//2]
                #         sampled_new = sampled_tensordict[cfg.batch_size//2:]
                #         prb_e.update_tensordict_priority(sampled_expert)
                #         prb.update_tensordict_priority(sampled_new)
                #     else:
                #         prb.update_tensordict_priority(sampled_tensordict)
                t2.append(time() - tic)

                total_losses.append(loss.item())
                actor_losses.append(actor_loss.item())
                q_losses.append(q_loss.item())
                alpha_losses.append(alpha_loss.item())
                alphas.append(loss_td["alpha"].item())
                entropies.append(loss_td["entropy"].item())
                # feedback_model_losses.append(fd_model_loss)

        metrics = {
            "collected_frames": collected_frames,
            "collected_traj": num_trajs,
            "time": time_stamp,
        }
        if loss is not None:
            metrics.update(
                {
                    "total_loss": np.mean(total_losses),
                    "actor_loss": np.mean(actor_losses),
                    "q_loss": np.mean(q_losses),
                    "alpha_loss": np.mean(alpha_losses),
                    "alpha": np.mean(alphas),
                    "entropy": np.mean(entropies),
                    # "feedback_model_loss": np.mean(feedback_model_losses),
                }
            )

        for key, value in metrics.items():
            logger.log_scalar(key, value, step=collected_frames)
            local_logger.log(x_axis="steps", y_axis=key, x_value=collected_frames, y_value=value, log_time=True)
            
        local_logger.save_log()

        torch.save(model.state_dict(), f"../Data/FINAL/{cfg.sub_name}/{exp}/{run_name}{stage}/{str(i+1)}.pth")
        if (i+1)>=stop_iter:
            if cfg.hf:
                save_training(model, feedback_model, prb, episode_success, all_hf, all_heu, loss_module, f'{cfg.sub_name}/saved_training/{run_name}{stage}', i+1, cfg.collector.frames_per_batch)
            collector.shutdown()
            return 0
        
    # if cfg.audio:
    #     stream.stop_streaming()
    print("end- ", round(time() - global_start_time, 4))


if __name__ == "__main__":    
    sac()
