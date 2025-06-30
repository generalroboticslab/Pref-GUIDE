train() {
    local SUBJECT_LIST=$1         # e.g., "1 10 13 15 37 39"
    local GAME_NAME=$2            # e.g., "hide_and_seek_1v1" (currently unused)
    local model_type=$3           # e.g., "../../vote_ref_old"
    local pretrain=$4
    local freeze=$5
    local activation=$6

    local available_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

    # Set num_gpus and max_jobs based on detected GPU count
    local num_gpus=$available_gpus
    local max_jobs=16  # Max parallel jobs = number of GPUs

    local job_count=0

    read -a SUBJECTS <<< "$SUBJECT_LIST"

    for subject in "${SUBJECTS[@]}"; do
        local gpu_id=$((job_count % num_gpus))

        echo "Launching subject $subject on GPU $gpu_id"

        CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
            --game="$GAME_NAME" \
            --num_epochs=50 \
            --model_type="$model_type" \
            --subject_id="$subject" \
            --pretrain="$pretrain" \
            --freeze_encoder="$freeze" \
            --use_activation="$activation" &

        ((job_count++))

        # Wait if running jobs exceed max_jobs
        while [ "$(jobs -r | wc -l)" -ge "$max_jobs" ]; do
            sleep 1
        done
    done

    wait
    echo -e "\e[32mAll jobs finished successfully.\e[0m"
}

train_ablation() {
    local SUBJECT_LIST=$1         # e.g., "1 10 13 15 37 39"
    local GAME_NAME=$2            # e.g., "hide_and_seek_1v1" (currently unused)
    local model_type=$3           # e.g., "../../vote_ref_old"
    local no_preference=$4
    local moving_window=$5

    local available_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

    # Set num_gpus and max_jobs based on detected GPU count
    local num_gpus=$available_gpus
    local max_jobs=16  # Max parallel jobs = number of GPUs

    local job_count=0

    read -a SUBJECTS <<< "$SUBJECT_LIST"

    for subject in "${SUBJECTS[@]}"; do
        local gpu_id=$((job_count % num_gpus))

        echo "Launching subject $subject on GPU $gpu_id"

        CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
            --game="$GAME_NAME" \
            --num_epochs=50 \
            --model_type="$model_type" \
            --subject_id="$subject" \
            --pretrain=True \
            --freeze_encoder=False \
            --use_activation=True \
            --no_preference_window="$no_preference" \ 
            --moving_window="$moving_window" &

        ((job_count++))

        # Wait if running jobs exceed max_jobs
        while [ "$(jobs -r | wc -l)" -ge "$max_jobs" ]; do
            sleep 1
        done
    done

    wait
    echo -e "\e[32mAll jobs finished successfully.\e[0m"
}



train_vote() {
    local SUBJECT_LIST=$1         # e.g., "1 10 13 15 37 39"
    local GAME_NAME=$2            # e.g., "hide_and_seek_1v1" (currently unused)    # e.g., "../../vote_ref_old"
    local pretrain=$3
    local freeze=$4
    local use_activation=$5
    local ref_model=$6

    local available_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

    # Set num_gpus and max_jobs based on detected GPU count
    local num_gpus=$available_gpus
    local max_jobs=15  # Max parallel jobs = number of GPUs

    local job_count=0

    read -a SUBJECTS <<< "$SUBJECT_LIST"

    for subject in "${SUBJECTS[@]}"; do
        local gpu_id=$((job_count % num_gpus))

        echo "Launching subject $subject on GPU $gpu_id"

        CUDA_VISIBLE_DEVICES=$gpu_id python train_vote_ref.py \
            --game="$GAME_NAME" \
            --model_type="vote_ref" \
            --num_epochs=200 \
            --subject_id="$subject" \
            --pretrain="$pretrain" \
            --freeze_encoder="$freeze" \
            --ref_model="$ref_model" \
            --use_activation="$use_activation" &


        ((job_count++))

        # Wait if running jobs exceed max_jobs
        while [ "$(jobs -r | wc -l)" -ge "$max_jobs" ]; do
            sleep 1
        done
    done

    wait
    echo -e "\e[32mAll jobs finished successfully.\e[0m"
}


train_vote() {
    local SUBJECT_LIST=$1         # e.g., "1 10 13 15 37 39"
    local GAME_NAME=$2            # e.g., "hide_and_seek_1v1" (currently unused)    # e.g., "../../vote_ref_old"
    local pretrain=$3
    local freeze=$4
    local use_activation=$5
    local ref_model=$6

    local available_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

    # Set num_gpus and max_jobs based on detected GPU count
    local num_gpus=$available_gpus
    local max_jobs=15  # Max parallel jobs = number of GPUs

    local job_count=0

    read -a SUBJECTS <<< "$SUBJECT_LIST"

    for subject in "${SUBJECTS[@]}"; do
        local gpu_id=$((job_count % num_gpus))

        echo "Launching subject $subject on GPU $gpu_id"

        CUDA_VISIBLE_DEVICES=$gpu_id python train_vote_ref_total.py \
            --game="$GAME_NAME" \
            --model_type="vote_ref" \
            --num_epochs=200 \
            --subject_id="$subject" \
            --pretrain="$pretrain" \
            --freeze_encoder="$freeze" \
            --ref_model="$ref_model" \
            --use_activation="$use_activation" &


        ((job_count++))

        # Wait if running jobs exceed max_jobs
        while [ "$(jobs -r | wc -l)" -ge "$max_jobs" ]; do
            sleep 1
        done
    done

    wait
    echo -e "\e[32mAll jobs finished successfully.\e[0m"
}
