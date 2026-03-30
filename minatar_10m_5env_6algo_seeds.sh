#!/bin/bash
#SBATCH --job-name=minatar-10m-5env-6algo
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=4
#SBATCH --ntasks=1
#SBATCH --array=0-59%60
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mem=32G
#SBATCH --time=72:00:00

################################################################################
# MinAtar 10M: 4 envs × 6 algos (MoG / DQN / C51 / QR-DQN / IQN / FQF) × 5 seeds = 120 tasks
#
# W&B: MinAtar_10M + one algo tag per run
#
# Mapping: TID 0–119 → env_idx = TID/30, rem = TID%30 → algo = rem/5, seed = rem%5
#
# Submit: sbatch minatar_10m_5env_6algo_seeds.sh
################################################################################

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

export WANDB_PROJECT="${WANDB_PROJECT:-Deep-CVI-Experiments}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

EXPERIMENT_TAG="MinAtar_10M"

# Which algorithm indices to actually run (0=MoG 1=DQN 2=C51 3=QR-DQN 4=IQN 5=FQF).
# Example — only distributional: ACTIVE_ALGOS=(3 4 5)
# Full sweep: ACTIVE_ALGOS=(0 1 2 3 4 5)
ACTIVE_ALGOS=(3 4 5)

TID="${SLURM_ARRAY_TASK_ID:?}"
ENV_IDX=$((TID / 30))
REM=$((TID % 30))
ALGO_IDX=$((REM / 5))
SEED_IDX=$((REM % 5))

_run=0
for _a in "${ACTIVE_ALGOS[@]}"; do
    if [[ "${_a}" -eq "${ALGO_IDX}" ]]; then _run=1; break; fi
done
if [[ "${_run}" -eq 0 ]]; then
    echo "Skipping TID=${TID} (algo_idx=${ALGO_IDX} not in ACTIVE_ALGOS=${ACTIVE_ALGOS[*]})"
    exit 0
fi

ENV_IDS=(
    "Asterix-MinAtar"
    "Breakout-MinAtar"
    "Freeway-MinAtar"
    "SpaceInvaders-MinAtar"
)

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
    1) EXTRA=() ;;
    2) EXTRA=(--n-atoms 51 --v-min 0 --v-max 500) ;;
    3) EXTRA=(--num-atoms 200 --kappa 1.0) ;;
    4) EXTRA=(--num-tau-samples 32 --num-tau-prime-samples 32 --num-quantile-samples 32 --quantile-embedding-dim 64 --kappa 1.0) ;;
    5) EXTRA=(--num-atoms 32 --num-tau-samples 32 --num-tau-prime-samples 32 --num-quantile-samples 32 --quantile-embedding-dim 64 --kappa 1.0 --ent-coef 0.001 --fqf-lr-factor 1e-6) ;;
    *) echo "Invalid ALGO_IDX=${ALGO_IDX}" >&2; exit 1 ;;
esac

echo "=========================================="
echo "MinAtar 10M — 4 envs × 6 algos × 5 seeds (ACTIVE_ALGOS=${ACTIVE_ALGOS[*]})"
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
