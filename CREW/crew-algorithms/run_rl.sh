seeds=(45 46)

for seed in "${seeds[@]}"
do
    python crew_algorithms/ddpg envs=bowling collector.frames_per_batch=120 batch_size=120 hf=False feedback_model=False seed=$seed envs.time_scale=1 sub_name=rl_baselines
    python crew_algorithms/ddpg envs=find_treasure collector.frames_per_batch=240 batch_size=240 hf=False feedback_model=False seed=$seed envs.time_scale=1 sub_name=rl_baselines
    python crew_algorithms/ddpg envs=hide_and_seek_1v1 collector.frames_per_batch=240 batch_size=240 hf=False feedback_model=False seed=$seed envs.time_scale=1 sub_name=rl_baselines

    python crew_algorithms/ddpg envs=find_treasure collector.frames_per_batch=240 batch_size=240 hf=False feedback_model=False heuristic_feedback=True seed=$seed envs.time_scale=1 sub_name=rl_baselines
    python crew_algorithms/ddpg envs=hide_and_seek_1v1 collector.frames_per_batch=240 batch_size=240 hf=False feedback_model=False heuristic_feedback=True seed=$seed envs.time_scale=1 sub_name=rl_baselines

    python crew_algorithms/sac envs=bowling collector.frames_per_batch=120 batch_size=120 hf=False feedback_model=False seed=$seed envs.time_scale=1 sub_name=rl_baselines
    python crew_algorithms/sac envs=find_treasure collector.frames_per_batch=240 batch_size=240 hf=False feedback_model=False seed=$seed envs.time_scale=1 sub_name=rl_baselines
    python crew_algorithms/sac envs=hide_and_seek_1v1 collector.frames_per_batch=240 batch_size=240 hf=False feedback_model=False seed=$seed envs.time_scale=1 sub_name=rl_baselines
done