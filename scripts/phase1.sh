set -ex
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
seeds=(342 4827 1215 98 8128 828)
source scripts/functions_phase1.sh

for MODEL_SEED in "${seeds[@]}";
do
    MODEL_ID="roberta_large"
    MODEL_NAME="roberta-large"
    # MODEL_ID="opt350"
    # MODEL_NAME="facebook/opt-350m"

    for TASK_NAME in mnli sst2 snli; do
        glue_baseline
    done
    squad_baseline
    race_baseline
done

# create h_score and amgbiuous scores using the outputs of the above runs
python create_scores.py --tasks mnli,snli,sst2 --models $MODEL_ID --outputs_path results_phase1 
python create_scores_squad.py --models $MODEL_ID --outputs_path results_phase1 
python create_scores_race.py --models $MODEL_ID --outputs_path results_phase1 