#!/bin/bash
#SBATCH --job-name=craftax-50m-6algo
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=4
#SBATCH --ntasks=1
#SBATCH --array=0-29%30
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mem=32G
#SBATCH --time=120:00:00

################################################################################
# Craftax Classic 50M: 6 algos (MoG / DQN / C51 / QR-DQN / IQN / FQF) × 5 seeds = 30 tasks
#
# W&B: Craftax_50M + MoG | dqn | C51 | QR-DQN | IQN | FQF
#
# Mapping: TID 0–29 → algo = TID/5, seed = TID%5
#
# Submit: sbatch craftax_6algo_seeds.sh
################################################################################

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

export WANDB_PROJECT="${WANDB_PROJECT:-Deep-CVI-Experiments}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

EXPERIMENT_TAG="Craftax_50M"

TID="${SLURM_ARRAY_TASK_ID:?}"
ALGO_IDX=$((TID / 5))
SEED_IDX=$((TID % 5))

MODULES=(
    cleanrl.cvi_dqn_nocollapse_jax
    cleanrl.dqn_gymnax_jax
    cleanrl.c51_craftax_jax
    cleanrl.qrdqn_gymnax_jax
    cleanrl.iqn_gymnax_jax
    cleanrl.fqf_gymnax_jax
)
ALGO_WANDB_TAGS=(MoG dqn C51 "QR-DQN" IQN FQF)
SEEDS=(7 23 42 314 543)

ENV_ID="${ENV_ID:-Craftax-Classic-Symbolic-v1}"
MODULE="${MODULES[$ALGO_IDX]}"
WANDB_TAG_ALGO="${ALGO_WANDB_TAGS[$ALGO_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
WANDB_TAGS="${EXPERIMENT_TAG},${WANDB_TAG_ALGO}"

COMMON=(
    --env-id "${ENV_ID}"
    --total-timesteps 50000000
    --learning-rate 0.0001
    --batch-size 512
    --end-e 0.05
    --seed "${SEED}"
    --exploration-fraction 0.2
    --buffer-size 500000
    --target-network-frequency 0
    --tau 0.005
    --num-envs 64
    --utd-ratio 0.1
    --learning-starts 10000
    --hidden1 256
    --hidden2 256
    --hidden3 128
    --log-interval 10000
    --track
    --wandb-project-name "${WANDB_PROJECT}"
    --wandb-tags "${WANDB_TAGS}"
)

case "${ALGO_IDX}" in
    0) EXTRA=(--num-omega-samples 128 --omega-max 1.0 --num-components 8) ;;
    1) EXTRA=() ;;
    2) EXTRA=(--n-atoms 51 --v-min 0 --v-max 22) ;;
    3) EXTRA=(--num-atoms 200 --kappa 1.0) ;;
    4) EXTRA=(--num-tau-samples 32 --num-tau-prime-samples 32 --num-quantile-samples 32 --quantile-embedding-dim 64 --kappa 1.0) ;;
    5) EXTRA=(--num-atoms 32 --num-tau-samples 32 --num-tau-prime-samples 32 --num-quantile-samples 32 --quantile-embedding-dim 64 --kappa 1.0 --ent-coef 0.001 --fqf-lr-factor 1e-6) ;;
    *) echo "Invalid ALGO_IDX=${ALGO_IDX}" >&2; exit 1 ;;
esac

echo "=========================================="
echo "Craftax 50M — 6 algos × 5 seeds"
echo "=========================================="
echo "Task:     ${TID} (algo ${ALGO_IDX}, seed slot ${SEED_IDX})"
echo "Env:      ${ENV_ID}"
echo "Module:   ${MODULE}"
echo "Seed:     ${SEED}"
echo "W&B tags: ${WANDB_TAGS}"
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Host:     $(hostname)"
echo "GPU:      ${CUDA_VISIBLE_DEVICES:-}"
echo "Start:    $(date)"
echo "=========================================="

srun uv run --no-sync python -m "${MODULE}" "${COMMON[@]}" "${EXTRA[@]}"

echo "Task ${TID} completed"
echo "End:      $(date)"
echo "=========================================="
