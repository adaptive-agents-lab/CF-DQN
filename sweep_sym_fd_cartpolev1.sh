#!/bin/bash
#SBATCH --job-name=sym-fd-sweep
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=4
#SBATCH --ntasks=1
#SBATCH --array=0-59%10          # 60 total runs (5 seeds × 4 n_collapse_pairs × 3 buffer_sizes), 10 parallel
#SBATCH --output=slurm/logs/sym-fd-sweep-cartpole/sweep_%A_%a.out
#SBATCH --mem-per-cpu=12G
#SBATCH --time=2:00:00           # CartPole 500k steps ≈ 5 min per run

################################################################################
# SymFD Sensitivity Sweep — CartPole-v1
################################################################################
# Sweeps over:  n_collapse_pairs ∈ {3,5,7,10}  ×  buffer_size ∈ {10k,25k,50k}  ×  5 seeds
# Total: 60 runs with grid method (every combination runs exactly once).
#
# INSTRUCTIONS:
# 1. Initialize sweep:  wandb sweep sweep_sym_fd_cartpolev1.yaml
# 2. Copy the sweep ID from the terminal output
# 3. Paste it in SWEEP_ID below
# 4. Submit:             sbatch sweep_sym_fd_cartpolev1.sh
#
# AGGREGATION (after sweep completes):
#   In W&B → Sweeps → SymFD-Sensitivity-CartPole-v1:
#     • Parallel coordinates plot: group by n_collapse_pairs & buffer_size
#     • Or use the API to compute mean±std per config (see analyze_sweep.py below)
################################################################################

# ===== EDIT THIS LINE WITH YOUR SWEEP ID =====
SWEEP_ID="${SWEEP_ID:-fatty_data/Deep-CVI-Experiments/538efz7i}"

# ==============================================

mkdir -p slurm/logs/sym-fd-sweep-cartpole/

echo "=========================================="
echo "SymFD Sensitivity Sweep — CartPole-v1"
echo "=========================================="
echo "Sweep ID:    ${SWEEP_ID}"
echo "Agent ID:    ${SLURM_ARRAY_TASK_ID}"
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Host:        $(hostname)"
echo "GPU:         ${CUDA_VISIBLE_DEVICES}"
echo "Start time:  $(date)"
echo "=========================================="

# Each SLURM task runs exactly 1 experiment then exits.
srun uv run wandb agent --count 1 ${SWEEP_ID}

echo "=========================================="
echo "Agent ${SLURM_ARRAY_TASK_ID} completed"
echo "End time: $(date)"
echo "=========================================="
