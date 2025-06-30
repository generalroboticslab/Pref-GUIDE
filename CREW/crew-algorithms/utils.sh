run_ddpg() {
    local SUBJECT_LIST=$1           # e.g., "1 10 13 15 37 39"
    local GAME_NAME=$2              # e.g., "hide_and_seek_1v1"
    local BASE_FEEDBACK_PATH=$3     # e.g., "../../vote_ref_old"

    local available_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

    # Set num_gpus and max_jobs based on detected GPU count
    if [ "$available_gpus" -eq 8 ]; then
        local max_jobs=8
    else
        local max_jobs=5
    fi

    local num_gpus=$available_gpus
    local job_count=0

    read -a SUBJECTS <<< "$SUBJECT_LIST"

    for subject in "${SUBJECTS[@]}"; do

        local gpu_id=$((job_count % num_gpus))

        DISPLAY=:0 CUDA_VISIBLE_DEVICES=$gpu_id python crew_algorithms/ddpg \
            envs=$GAME_NAME \
            feedback_model=True \
            feedback_model_path="${BASE_FEEDBACK_PATH}/${GAME_NAME}/subject_${subject}.pt" \
            continue_training="${subject}_${GAME_NAME}_guide" \
            &

        ((job_count++))

        # Wait if running jobs exceed max_jobs
        while [ "$(jobs -r | wc -l)" -ge $max_jobs ]; do
            sleep 1
        done
    done

    wait

    echo -e "\e[32mAll jobs finished successfully.\e[0m"
}

run_ddpg_from_scratch() {
    local SUBJECT_LIST=$1           # e.g., "1 10 13 15 37 39"
    local GAME_NAME=$2              # e.g., "hide_and_seek_1v1"
    local BASE_FEEDBACK_PATH=$3     # e.g., "../../vote_ref_old"

    local available_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

    # Set num_gpus and max_jobs based on detected GPU count
    if [ "$available_gpus" -eq 8 ]; then
        local max_jobs=8
    else
        local max_jobs=5
    fi

    local num_gpus=$available_gpus
    local job_count=0

    read -a SUBJECTS <<< "$SUBJECT_LIST"

    for subject in "${SUBJECTS[@]}"; do

        local gpu_id=$((job_count % num_gpus))

        DISPLAY=:0 CUDA_VISIBLE_DEVICES=$gpu_id python crew_algorithms/ddpg \
            envs=$GAME_NAME \
            feedback_model=True \
            feedback_model_path="${BASE_FEEDBACK_PATH}/${GAME_NAME}/subject_${subject}.pt" \
            &

        ((job_count++))

        # Wait if running jobs exceed max_jobs
        while [ "$(jobs -r | wc -l)" -ge $max_jobs ]; do
            sleep 1
        done
    done

    wait

    echo -e "\e[32mAll jobs finished successfully.\e[0m"
}


run_ddpg_baseline1() {
    local SEED_LIST=$1           # e.g., "1 10 13 15 37 39"
    local GAME_NAME=$2              # e.g., "hide_and_seek_1v1"
    local HEURISTIC=$3

    local available_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

    # Set num_gpus and max_jobs based on detected GPU count
    if [ "$available_gpus" -eq 8 ]; then
        local max_jobs=8
    else
        local max_jobs=5
    fi

    local num_gpus=$available_gpus

    local job_count=0

    read -a SEEDS <<< "$SEED_LIST"

    for seed in "${SEEDS[@]}"; do

        local gpu_id=$((job_count % num_gpus))

        DISPLAY=:0 CUDA_VISIBLE_DEVICES=$gpu_id python crew_algorithms/ddpg \
            envs=$GAME_NAME \
            heuristic_feedback=$HEURISTIC \
            seed=$seed &

        ((job_count++))

        # Wait if running jobs exceed max_jobs
        while [ "$(jobs -r | wc -l)" -ge $max_jobs ]; do
            sleep 1
        done
    done

    wait

    echo -e "\e[32mAll jobs finished successfully.\e[0m"
}

eval_ddpg(){
    local SUBJECT_LIST=$1           # e.g., "1 10 13 15 37 39"
    local GAME_NAME=$2              # e.g., "hide_and_seek_1v1"
    local eval_path=$3

    local available_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

    # Set num_gpus and max_jobs based on detected GPU count
    if [ "$available_gpus" -eq 8 ]; then
        local max_jobs=5
    else
        local max_jobs=5
    fi

    local num_gpus=$available_gpus
    local job_count=0

    read -a SUBJECTS <<< "$SUBJECT_LIST"

    for subject in "${SUBJECTS[@]}"; do

        local gpu_id=$((job_count % num_gpus))

        DISPLAY=:0 CUDA_VISIBLE_DEVICES=$gpu_id python crew_algorithms/ddpg/eval.py  \
            envs=$GAME_NAME \
            exp_path="ddpg_cont/${GAME_NAME}/${eval_path}/subject_${subject}"\
            seed=47 \
            &

        ((job_count++))

        # Wait if running jobs exceed max_jobs
        while [ "$(jobs -r | wc -l)" -ge $max_jobs ]; do
            sleep 1
        done
    done

    wait

    echo -e "\e[32mAll jobs finished successfully.\e[0m"

}

eval_ddpg_local(){
    local SUBJECT_LIST=$1           # e.g., "1 10 13 15 37 39"
    local GAME_NAME=$2              # e.g., "hide_and_seek_1v1"
    local eval_path=$3


    local max_jobs=5
    local num_gpus=1
    local job_count=0

    read -a SUBJECTS <<< "$SUBJECT_LIST"

    for subject in "${SUBJECTS[@]}"; do

        local gpu_id=$((job_count % num_gpus))

        CUDA_VISIBLE_DEVICES=$gpu_id python crew_algorithms/ddpg/eval.py  \
            envs=$GAME_NAME \
            exp_path="ddpg_cont/${GAME_NAME}/${eval_path}/subject_${subject}"\
            seed=47 \
            &

        ((job_count++))

        # Wait if running jobs exceed max_jobs
        while [ "$(jobs -r | wc -l)" -ge $max_jobs ]; do
            sleep 1
        done
    done

    wait

    echo -e "\e[32mAll jobs finished successfully.\e[0m"

}

run_ddpg_reproduce() {
    local SUBJECT_LIST=$1           # e.g., "1 10 13 15 37 39"
    local GAME_NAME=$2              # e.g., "hide_and_seek_1v1"
    local BASE_FEEDBACK_PATH=$3     # e.g., "../../vote_ref_old"

    local available_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

    # Set num_gpus and max_jobs based on detected GPU count
    if [ "$available_gpus" -eq 8 ]; then
        local max_jobs=8
    else
        local max_jobs=4
    fi

    local num_gpus=$available_gpus
    local job_count=0

    read -a SUBJECTS <<< "$SUBJECT_LIST"

    for subject in "${SUBJECTS[@]}"; do

        local gpu_id=$((job_count % num_gpus))

        DISPLAY=:0 CUDA_VISIBLE_DEVICES=$gpu_id python crew_algorithms/ddpg \
            envs=$GAME_NAME \
            feedback_model=True \
            feedback_model_path="${BASE_FEEDBACK_PATH}/${GAME_NAME}/subject_${subject}.pt" \
            continue_training="${subject}_${GAME_NAME}_guide" \
            &

        ((job_count++))

        # Wait if running jobs exceed max_jobs
        while [ "$(jobs -r | wc -l)" -ge $max_jobs ]; do
            sleep 1
        done
    done

    wait

    echo -e "\e[32mAll jobs finished successfully.\e[0m"
}


run_ddpg_baseline() {
    local SUBJECT_LIST=$1           # e.g., "1 10 13 15 37 39"
    local GAME_NAME=$2              # e.g., "hide_and_seek_1v1"
    local HEURISTIC=$3

    local available_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

    # Set num_gpus and max_jobs based on detected GPU count
    if [ "$available_gpus" -eq 8 ]; then
        local max_jobs=8
    else
        local max_jobs=5
    fi

    local num_gpus=$available_gpus

    local job_count=0

    read -a SUBJECTS <<< "$SUBJECT_LIST"

    for subject in "${SUBJECTS[@]}"; do

        local gpu_id=$((job_count % num_gpus))

        DISPLAY=:0 CUDA_VISIBLE_DEVICES=$gpu_id python crew_algorithms/ddpg \
            envs=$GAME_NAME \
            heuristic_feedback=$HEURISTIC \
            continue_training="${subject}_${GAME_NAME}_guide" \
            &

        ((job_count++))

        # Wait if running jobs exceed max_jobs
        while [ "$(jobs -r | wc -l)" -ge $max_jobs ]; do
            sleep 1
        done
    done

    wait

    echo -e "\e[32mAll jobs finished successfully.\e[0m"
}


run_ddpg_baseline_local() {
    local SEED_LIST=$1           # e.g., "1 10 13 15 37 39"
    local GAME_NAME=$2              # e.g., "hide_and_seek_1v1"
    local HEURISTIC=$3

    local available_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

    # Set num_gpus and max_jobs based on detected GPU count

    local max_jobs=2


    local num_gpus=$available_gpus

    local job_count=0

    read -a SEEDS <<< "$SEED_LIST"

    for seed in "${SEEDS[@]}"; do

        local gpu_id=$((job_count % num_gpus))

        python crew_algorithms/ddpg envs=$GAME_NAME heuristic_feedback=$HEURISTIC seed=$seed &

        ((job_count++))

        # Wait if running jobs exceed max_jobs
        while [ "$(jobs -r | wc -l)" -ge $max_jobs ]; do
            sleep 1
        done
    done

    wait

    echo -e "\e[32mAll jobs finished successfully.\e[0m"
}