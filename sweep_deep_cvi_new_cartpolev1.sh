#!/bin/bash
#SBATCH --job-name=deep-cvi-sweep
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=4
#SBATCH --ntasks=1
#SBATCH --array=0-99%20          # Run 100 total experiments, max 20 at a time
#SBATCH --output=slurm/logs/new-cvi-sweep-cartpole/sweep_%A_%a.out
#SBATCH --mem-per-cpu=12G
#SBATCH --time=6:00:00           # Lowered to 4 hrs since CartPole runs very fast

################################################################################
# CVI-DQN Wandb Sweep - SLURM Parallel Execution
################################################################################
# INSTRUCTIONS:
# 1. Initialize sweep:        wandb sweep sweep.yaml
# 2. Copy the sweep ID from the terminal output
# 3. Edit SWEEP_ID below with your actual sweep ID
# 4. Submit job:              sbatch sweep_cvi.sh
################################################################################

# ===== EDIT THIS LINE WITH YOUR SWEEP ID =====
SWEEP_ID="${SWEEP_ID:-fatty_data/CF-DQN-cleanrl/p6os3tgt}"
# ==============================================

# Create log directory if it doesn't exist
mkdir -p slurm/logs/new-cvi-sweep-cartpole/

echo "=========================================="
echo "CVI-DQN Wandb Sweep Agent"
echo "=========================================="
echo "Sweep ID:    ${SWEEP_ID}"
echo "Agent ID:    ${SLURM_ARRAY_TASK_ID}"
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Host:        $(hostname)"
echo "GPU:         ${CUDA_VISIBLE_DEVICES}"
echo "Start time:  $(date)"
echo "=========================================="

# Run wandb agent (using uv to run the environment)
# The --count 1 flag tells this specific SLURM task to run exactly ONE experiment and then die.
# The array handles spawning the next tasks.
srun uv run wandb agent --count 1 ${SWEEP_ID}

echo "=========================================="
echo "Agent ${SLURM_ARRAY_TASK_ID} completed"
echo "End time: $(date)"
echo "=========================================="
