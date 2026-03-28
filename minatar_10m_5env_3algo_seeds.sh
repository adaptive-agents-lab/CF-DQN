#!/bin/bash
#SBATCH --job-name=minatar-10m-5env
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=4
#SBATCH --ntasks=1
#SBATCH --array=0-74%75
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mem=32G
#SBATCH --time=72:00:00

################################################################################
# MinAtar 10M steps: 5 envs × 3 algos (MoG / FFT / dqn) × 5 seeds = 75 tasks
#
# W&B tags per run:  MinAtar_10M  +  MoG | FFT | dqn
#
# Environments (gymnax ids):
#   Asterix, Breakout, Freeway, SpaceInvaders — MinAtar
#   Seaquest-MinAtar is NOT implemented in current gymnax (NotImplementedError);
#   fifth slot uses Pong-misc as the extra arcade env. Swap to Seaquest when supported.
#
# Mapping: TID 0–74 → env_idx = TID/15, rem = TID%15 → algo = rem/5, seed = rem%5
#
# Submit: sbatch minatar_10m_5env_3algo_seeds.sh
################################################################################

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

export WANDB_PROJECT="${WANDB_PROJECT:-Deep-CVI-Experiments}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

EXPERIMENT_TAG="MinAtar_10M"

TID="${SLURM_ARRAY_TASK_ID:?}"
ENV_IDX=$((TID / 15))
REM=$((TID % 15))
ALGO_IDX=$((REM / 5))
SEED_IDX=$((REM % 5))

ENV_IDS=(
    "Asterix-MinAtar"
    "Breakout-MinAtar"
    "Freeway-MinAtar"
    "SpaceInvaders-MinAtar"
    "Pong-misc"
)

MODULES=(cleanrl.cvi_dqn_nocollapse_jax cleanrl.cvi_dqn_jax cleanrl.dqn_gymnax_jax)
ALGO_WANDB_TAGS=(MoG FFT dqn)
SEEDS=(7 23 42 314 543)

ENV_ID="${ENV_IDS[$ENV_IDX]}"
MODULE="${MODULES[$ALGO_IDX]}"
WANDB_TAG_ALGO="${ALGO_WANDB_TAGS[$ALGO_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
WANDB_TAGS="${EXPERIMENT_TAG},${WANDB_TAG_ALGO}"

COMMON=(
    --env-id "${ENV_ID}"
    --total-timesteps 10000000
    --learning-rate 0.00025
    --batch-size 512
    --end-e 0.01
    --seed "${SEED}"
    --exploration-fraction 0.3
    --buffer-size 500000
    --target-network-frequency 0
    --tau 0.005
    --num-envs 64
    --utd-ratio 0.1
    --track
    --wandb-project-name "${WANDB_PROJECT}"
    --wandb-tags "${WANDB_TAGS}"
)

case "${ALGO_IDX}" in
    0) EXTRA=(--num-omega-samples 128 --omega-max 1.0 --num-components 8) ;;
    1) EXTRA=(--K 128 --w 1.0 --q-min 0 --q-max 500) ;;
    2) EXTRA=() ;;
    *) echo "Invalid ALGO_IDX=${ALGO_IDX}" >&2; exit 1 ;;
esac

echo "=========================================="
echo "MinAtar 10M — 5 envs × 3 algos × 5 seeds"
echo "=========================================="
echo "Task:     ${TID} (env ${ENV_IDX}, algo ${ALGO_IDX}, seed slot ${SEED_IDX})"
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
