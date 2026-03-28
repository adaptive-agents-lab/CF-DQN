#!/bin/bash
#SBATCH --job-name=breakout-minatar-3algo
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=4
#SBATCH --ntasks=1
#SBATCH --array=0-14%15
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mem=16G
#SBATCH --time=48:00:00

################################################################################
# Breakout-MinAtar: MoG (nocollapse) vs FFT (cvi_dqn_jax) vs dqn_gymnax_jax
# 3 algos × 5 seeds = 15 tasks.  task_id → algo_idx = task/5, seed_idx = task%5
#
# Submit: sbatch breakout_minatar_3algo_seeds.sh
# uv: `uv sync --frozen` on login once; `--no-sync` avoids per-task resync.
################################################################################

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

export WANDB_PROJECT="${WANDB_PROJECT:-Deep-CVI-Experiments}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

TID="${SLURM_ARRAY_TASK_ID:?}"
ALGO_IDX=$((TID / 5))
SEED_IDX=$((TID % 5))

# One entry per algorithm (index 0..2)
MODULES=(cleanrl.cvi_dqn_nocollapse_jax cleanrl.cvi_dqn_jax cleanrl.dqn_gymnax_jax)
WANDB_TAGS=(MoG FFT dqn)

# Reproducibility: five fixed seeds (all ≠ 0, 1)
SEEDS=(7 23 42 314 543)

MODULE="${MODULES[$ALGO_IDX]}"
WANDB_TAG="${WANDB_TAGS[$ALGO_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

# Shared by all three runners
COMMON=(
    --env-id Breakout-MinAtar
    --total-timesteps 3000000
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
    --wandb-tags "${WANDB_TAG}"
)

case "${ALGO_IDX}" in
    0) EXTRA=(--num-omega-samples 128 --omega-max 1.0 --num-components 8) ;;
    1) EXTRA=(--K 128 --w 1.0 --q-min 0 --q-max 150) ;;
    2) EXTRA=() ;;
    *) echo "Invalid ALGO_IDX=${ALGO_IDX}" >&2; exit 1 ;;
esac

echo "=========================================="
echo "Breakout-MinAtar — 3-algo × 5 seeds"
echo "=========================================="
echo "Task:     ${TID} (algo ${ALGO_IDX}, seed slot ${SEED_IDX})"
echo "Module:   ${MODULE}"
echo "Seed:     ${SEED}"
echo "W&B tag:  ${WANDB_TAG}"
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Host:     $(hostname)"
echo "GPU:      ${CUDA_VISIBLE_DEVICES:-}"
echo "Start:    $(date)"
echo "=========================================="

srun uv run --no-sync python -m "${MODULE}" "${COMMON[@]}" "${EXTRA[@]}"

echo "Task ${TID} completed"
echo "End:      $(date)"
echo "=========================================="
