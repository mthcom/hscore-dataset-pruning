glue_pruned() {
    OUTPUT_DIR="./results_phase2/${PRUNING_METHOD}/remove_${REMOVE_STR}/${MODEL_ID}/TASK:${TASK_NAME}_SEEDS:${MODEL_SEED}"
    if [ -f "$OUTPUT_DIR/eval_results.json" ]; then
        return
    fi

    if [ "$TASK_NAME" = "snli" ]; then
        DATA_ARG="--dataset_name"
    else
        DATA_ARG="--task_name"
    fi

    python run_glue.py --do_train --overwrite_output_dir --do_eval --max_seq_length 128 --per_device_eval_batch_size 128  --per_device_train_batch_size 32 --learning_rate $LR --num_train_epochs 3 --fp16 --model_name_or_path $MODEL_NAME --output_dir $OUTPUT_DIR $DATA_ARG $TASK_NAME --seed $MODEL_SEED --data_seed $MODEL_SEED --sampler_seed $MODEL_SEED --evaluation_strategy epoch --save_strategy no --pruning_method $PRUNING_METHOD --rewards_file "./scores/hscore/${MODEL_ID}/${TASK_NAME}" --scores_to_remove $REMOVE_STR --epochs_used 3 --runs_used 6 --label_probs_file "./scores/ambiguous/${MODEL_ID}/${TASK_NAME}"
}

squad_pruned() {
    OUTPUT_DIR="./results_phase2/${PRUNING_METHOD}/remove_${REMOVE_STR}/${MODEL_ID}/TASK:squad_v2_SEEDS:${MODEL_SEED}"
    if [ -f "$OUTPUT_DIR/eval_results.json" ]; then
        return
    fi
    python run_qa.py --do_train --evaluation_strategy epoch --overwrite_output_dir --do_eval --max_seq_length 384 --doc_stride 128 --save_strategy no --per_device_train_batch_size 12 --learning_rate 3e-05 --num_train_epochs 2 --fp16 --model_name_or_path $MODEL_NAME --output_dir $OUTPUT_DIR --dataset_name squad_v2 --seed $MODEL_SEED --pruning_method $PRUNING_METHOD --rewards_file "./scores/hscore/${MODEL_ID}/squad_v2" --scores_to_remove $REMOVE_STR --epochs_used 2 --runs_used 6 --label_probs_file "./scores/ambiguous/${MODEL_ID}/squad_v2"
}

race_pruned() {
    OUTPUT_DIR="./results_phase2/${PRUNING_METHOD}/remove_${REMOVE_STR}/${MODEL_ID}/TASK:race_SEEDS:${MODEL_SEED}"
    if [ -f "$OUTPUT_DIR/eval_results.json" ]; then
        return
    fi
    python run_race.py --model_name_or_path $MODEL_NAME --dataset_name race --dataset_config_name all --do_train --do_eval --per_device_train_batch_size 2 --learning_rate 1e-5 --num_train_epochs 3 --gradient_accumulation_steps 8 --output_dir $OUTPUT_DIR --overwrite_output --pad_to_max_length --evaluation_strategy epoch --save_strategy 'no' --fp16 --sampler_seed $MODEL_SEED --seed $MODEL_SEED --data_seed $MODEL_SEED --pruning_method $PRUNING_METHOD --rewards_file "./scores/hscore/${MODEL_ID}/race" --scores_to_remove $REMOVE_STR --epochs_used 3 --runs_used 6 --label_probs_file "./scores/ambiguous/${MODEL_ID}/race"
}

