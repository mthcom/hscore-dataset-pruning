set -ex
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
seeds=(6666)

source scripts/functions_phase2.sh

removed_scores=(06 016 0126 01236 0156 012346 012356)
nb_ablations=${#removed_scores[@]}


for MODEL_SEED in "${seeds[@]}";
do
    for ((i=0; i<$nb_ablations; i++)); do
        LR=1e-5
        MODEL_ID="roberta_large"
        MODEL_NAME="roberta-large"
        
        # LR=2e-5
        # MODEL_ID="opt350"
        # MODEL_NAME="facebook/opt-350m"

        REMOVE_STR=${removed_scores[$i]}

        for PRUNING_METHOD in hscore ambiguous random; do
            for TASK_NAME in mnli sst2 snli; do
                glue_pruned
            done
            squad_pruned
            race_pruned
        done
    done
done
