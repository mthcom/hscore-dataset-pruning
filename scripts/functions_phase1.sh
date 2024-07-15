glue_baseline() {
    OUTPUT_DIR="./results_phase1/${MODEL_ID}/TASK:${TASK_NAME}_SEED:${MODEL_SEED}"
    if [ -f "$OUTPUT_DIR/eval_results.json" ]; then
        return
    fi

    if [ "$TASK_NAME" = "snli" ]; then
        DATA_ARG="--dataset_name"
    else
        DATA_ARG="--task_name"
    fi

    python run_glue.py --do_train --overwrite_output_dir --do_eval --max_seq_length 128  --per_device_train_batch_size 32 --learning_rate 2e-05 --num_train_epochs 3 --fp16 --model_name_or_path $MODEL_NAME --output_dir $OUTPUT_DIR --seed $MODEL_SEED --data_seed $MODEL_SEED --sampler_seed $MODEL_SEED --evaluation_strategy epoch --save_strategy no $DATA_ARG $TASK_NAME
}

squad_baseline() {
    OUTPUT_DIR="./results_phase1/${MODEL_ID}/TASK:squad_v2_SEED:${MODEL_SEED}"
    if [ -f "$OUTPUT_DIR/eval_results.json" ]; then
        return
    fi
    python run_qa.py --do_train --evaluation_strategy epoch --overwrite_output_dir --do_eval --max_seq_length 384 --doc_stride 128 --save_strategy no --per_device_train_batch_size 12 --learning_rate 3e-05 --num_train_epochs 2 --fp16 --model_name_or_path $MODEL_NAME --output_dir $OUTPUT_DIR --dataset_name squad_v2 --seed $MODEL_SEED --sampler_seed $MODEL_SEED --data_seed $MODEL_SEED
}

race_baseline() {
    OUTPUT_DIR="./results_phase1/${MODEL_ID}/TASK:race_SEEDS:${MODEL_SEED}"
    if [ -f "$OUTPUT_DIR/eval_results.json" ]; then
        return
    fi
    python run_race.py --model_name_or_path $MODEL_NAME --dataset_name race --dataset_config_name all --do_train --do_eval --per_device_train_batch_size 2 --learning_rate 1e-5 --num_train_epochs 3 --gradient_accumulation_steps 8 --output_dir $OUTPUT_DIR --overwrite_output --pad_to_max_length --evaluation_strategy epoch --save_strategy 'no' --fp16 --sampler_seed $MODEL_SEED --seed $MODEL_SEED --data_seed $MODEL_SEED
}
