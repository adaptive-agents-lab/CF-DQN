#!/bin/bash
#SBATCH --job-name=cvi-vs-c51-atari
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=4      # 4 CPUs per GPU: env stepping is CPU-bound, keeps GPU fed
#SBATCH --ntasks=1
#SBATCH --array=0-11%12       # 12 independent tasks, each gets its own GPU (12 GPUs in parallel)
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mem-per-cpu=8G      # 4 CPUs × 8G = 32G RAM per task (replay buffer ~14G + overhead)
#SBATCH --time=24:00:00

################################################################################
# CVI-DQN vs C51 Atari Benchmark
################################################################################
# 2 algorithms × 2 envs × 3 seeds = 12 total runs (all parallel)
#
# Array index mapping:
#   0- 5 → c51_atari.py     (0-2: Pong seeds 1-3 | 3-5: Breakout seeds 1-3)
#   6-11 → cvi_dqn_atari.py (6-8: Pong seeds 1-3 | 9-11: Breakout seeds 1-3)
#
# Submit: sbatch cvi_vs_c51_atari.sh
################################################################################

MODULES=(
    "cleanrl.c51_atari"
    "cleanrl.c51_atari"
    "cleanrl.c51_atari"
    "cleanrl.c51_atari"
    "cleanrl.c51_atari"
    "cleanrl.c51_atari"
    "cleanrl.cvi_dqn_atari"
    "cleanrl.cvi_dqn_atari"
    "cleanrl.cvi_dqn_atari"
    "cleanrl.cvi_dqn_atari"
    "cleanrl.cvi_dqn_atari"
    "cleanrl.cvi_dqn_atari"
)

ENV_IDS=(
    "PongNoFrameskip-v4"
    "PongNoFrameskip-v4"
    "PongNoFrameskip-v4"
    "BreakoutNoFrameskip-v4"
    "BreakoutNoFrameskip-v4"
    "BreakoutNoFrameskip-v4"
    "PongNoFrameskip-v4"
    "PongNoFrameskip-v4"
    "PongNoFrameskip-v4"
    "BreakoutNoFrameskip-v4"
    "BreakoutNoFrameskip-v4"
    "BreakoutNoFrameskip-v4"
)

SEEDS=(1 2 3 1 2 3 1 2 3 1 2 3)

MODULE=${MODULES[$SLURM_ARRAY_TASK_ID]}
ENV_ID=${ENV_IDS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "CVI-DQN vs C51 — Atari Benchmark"
echo "=========================================="
echo "Task:     ${SLURM_ARRAY_TASK_ID} / 11"
echo "Module:   ${MODULE}"
echo "Env:      ${ENV_ID}"
echo "Seed:     ${SEED}"
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Host:     $(hostname)"
echo "GPU:      ${CUDA_VISIBLE_DEVICES}"
echo "Start:    $(date)"
echo "=========================================="


# Had to uv sync --frozen from login node to create a stable .venv
# Then I had to use --no-sync here to avoid the overhead of syncing the entire repo for each task otherwise they would access the same .venv and cause conflicts.

srun uv run --no-sync python -m ${MODULE} \
    --env-id ${ENV_ID} \
    --seed ${SEED} \
    --total-timesteps 10000000 \
    --wandb-project-name Deep-CVI-Experiments \
    --track

echo "Task ${SLURM_ARRAY_TASK_ID} completed"
echo "End:      $(date)"
echo "=========================================="
