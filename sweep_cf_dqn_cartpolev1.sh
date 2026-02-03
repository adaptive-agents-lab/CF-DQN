#!/bin/bash
#SBATCH --job-name=cf-dqn-sweep-cartpolev1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=4
#SBATCH --ntasks=1
#SBATCH --array=0-29%30
#SBATCH --output=slurm/logs/sweep_%A_%a.out
#SBATCH --mem-per-cpu=12G
#SBATCH --time=6:00:00

################################################################################
# CF-DQN Wandb Sweep - SLURM Parallel Execution
################################################################################
# This script runs multiple parallel wandb agents for hyperparameter sweeps
#
# INSTRUCTIONS:
# 1. Initialize sweep:        wandb sweep sweep_cf_dqn.yaml
# 2. Copy the sweep ID from the output
# 3. Edit SWEEP_ID below with your actual sweep ID
# 4. Submit job:              sbatch sweep_cf_dqn_cartpolev1.sh
#
# CONFIGURATION:
# - Total runs: 180 (3 seeds × 3 n_frequencies × 5 freq_max × 4 penalty_weight)
# - Parallel agents: 30 (adjust --array=0-29%30 as needed)
# - Time per run: ~10-15 minutes
# - Total sweep time: ~1-2 hours with 30 agents
################################################################################

# ===== EDIT THIS LINE WITH YOUR SWEEP ID =====
SWEEP_ID="${SWEEP_ID:-fatty_data/CF-DQN-cleanrl/v5ehvm46}"
# ==============================================

# Validate sweep ID
if [[ "$SWEEP_ID" == *"YOUR_ENTITY"* ]]; then
    echo "ERROR: Please set SWEEP_ID before submitting!"
    echo "Run: wandb sweep sweep_cf_dqn.yaml"
    echo "Then edit sweep_cf_dqn_cartpolev1.sh with the sweep ID"
    exit 1
fi

uv sync

# Create log directory if it doesn't exist
mkdir -p slurm/logs

echo "=========================================="
echo "CF-DQN Wandb Sweep Agent"
echo "=========================================="
echo "Sweep ID:    ${SWEEP_ID}"
echo "Agent:       ${SLURM_ARRAY_TASK_ID}"
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Host:        $(hostname)"
echo "GPU:         ${CUDA_VISIBLE_DEVICES}"
echo "Start time:  $(date)"
echo "=========================================="

# Run wandb agent
#srun wandb agent ${SWEEP_ID}

echo "=========================================="
echo "Agent ${SLURM_ARRAY_TASK_ID} completed"
echo "End time: $(date)"
echo "=========================================="
