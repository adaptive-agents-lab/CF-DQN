# CF-DQN Implementation Plan for CartPole

## Overview

This document outlines the implementation plan for CF-DQN (Characteristic Function Deep Q-Network), a distributional RL algorithm that operates in the frequency domain using characteristic functions instead of value distributions in the spatial domain.

**Goal**: Implement a working CF-DQN agent on CartPole-v1 as a stepping stone toward Atari environments.

**Core Idea**: Instead of learning Q(s,a) as a scalar or a distribution over atoms (C51), we learn Q_CF(s,a,ω) — the characteristic function of the return distribution evaluated at frequencies ω.

---

## Phase 1: Project Setup and File Structure

### Step 1.1: Create Directory Structure

Following cleanrl conventions, create files in the appropriate locations:

| File | Purpose |
|------|---------|
| `cleanrl_utils/cf.py` | CF operations: grid construction, interpolation, collapse |
| `cleanrl/cf_dqn.py` | Main training script for CartPole (single-file, self-contained) |
| `cleanrl_utils/evals/cf_dqn_eval.py` | Evaluation script (following C51 eval pattern) |

**Rationale**: 
- Utilities belong in `cleanrl_utils/` (like `buffers.py`, `atari_wrappers.py`)
- Algorithm scripts are single files in `cleanrl/`
- Eval scripts go in `cleanrl_utils/evals/`

### Step 1.2: Dependencies

No new dependencies required. Uses existing:
- PyTorch (tensors, autograd, complex numbers)
- Gymnasium (CartPole-v1)
- NumPy (utility operations)
- TensorBoard / WandB (logging)
- tyro (CLI argument parsing, cleanrl standard)

### Step 1.3: Import Structure for `cf_dqn.py`

Follow cleanrl's standard import pattern:

**Standard Library**:
- `os`, `random`, `time`
- `dataclasses.dataclass`

**Third Party**:
- `gymnasium as gym`
- `numpy as np`
- `torch`, `torch.nn`, `torch.optim`
- `tyro`
- `torch.utils.tensorboard.SummaryWriter`

**Local (cleanrl_utils)**:
- `from cleanrl_utils.buffers import ReplayBuffer`
- `from cleanrl_utils.cf import make_omega_grid, interpolate_cf_polar, collapse_cf_to_mean, reward_cf`

---

## Phase 2: CF Utilities Module (`cleanrl_utils/cf.py`)

Port the essential CF operations from `cvi_rl/cf/` to PyTorch tensors. This module will be imported as:
`from cleanrl_utils.cf import make_omega_grid, interpolate_cf_polar, collapse_cf_to_mean`

### Step 2.1: Frequency Grid Construction

Implement `make_omega_grid()` with **three_density_regions** strategy:
- Allocates 50% of points in inner 10% of frequency range (dense core)
- Allocates 30% of points in next 30% of range (medium density)
- Allocates 20% of points in outer 60% of range (sparse tails)

**Function signature**: `make_omega_grid(W, K, device) → torch.Tensor [K]`

**Note**: Hardcode `three_density_regions` strategy for simplicity. Can be parameterized later if needed.

### Step 2.2: CF Interpolation (Polar Method)

Implement `interpolate_cf_polar()` for evaluating CF at scaled frequencies γω:
1. Compute magnitude: |φ(ω)|
2. Compute unwrapped phase: unwrap(angle(φ(ω)))
3. Interpolate magnitude at target frequencies (linear interp)
4. Interpolate phase at target frequencies (linear interp)
5. Reconstruct: magnitude × exp(i × phase)

**Function signature**: `interpolate_cf_polar(target_ω, grid_ω, cf) → torch.Tensor [batch, n_actions, K]`

**Note**: Use `torch.angle()` and implement phase unwrapping via cumulative sum of wrapped differences. Must be batched and differentiable.

### Step 2.3: CF Collapse (Gaussian Method)

Implement `collapse_cf_to_mean()` to extract E[G] from φ(ω):
1. Select frequencies in range [-max_w, max_w] where signal is strong
2. Compute unwrapped phase of CF
3. Fit line: phase ≈ μ × ω (linear regression through origin)
4. Return μ as the estimated mean

**Function signature**: `collapse_cf_to_mean(omegas, cf, max_w) → torch.Tensor [batch, n_actions]`

**Batching**: Must handle shape [batch, n_actions, K] and return [batch, n_actions].

### Step 2.4: Reward CF Computation

Implement `reward_cf()` to compute exp(iωr) for immediate rewards:

**Function signature**: `reward_cf(omegas, rewards) → torch.Tensor [batch, K]`

**Note**: `rewards` has shape [batch, 1], output expands to [batch, K].

---

## Phase 3: Network Architecture

### Step 3.1: CF Q-Network Design

Design a network that outputs characteristic functions for each action, following cleanrl's C51 architecture pattern.

**Architecture Overview** (matching C51's `QNetwork` class):
1. **Shared Encoder**: Same hidden layer structure as C51
2. **CF Output**: Single linear layer outputting real and imaginary parts

**Layer Structure** (for CartPole, matching C51):
- Input: observation vector (dim = `env.single_observation_space.shape.prod()`)
- Hidden Layer 1: Linear(obs_dim, 120) + ReLU
- Hidden Layer 2: Linear(120, 84) + ReLU
- Output Layer: Linear(84, n_actions × K × 2)

**Rationale**: C51 uses `Linear(obs_dim, 120) → ReLU → Linear(120, 84) → ReLU → Linear(84, n_actions * n_atoms)`. We match this exactly, but output `n_actions × K × 2` instead of `n_actions × n_atoms` (×2 for real/imag).

**Output Reshaping**:
- Raw output shape: [batch, n_actions × K × 2]
- Reshape to: [batch, n_actions, K, 2]
- Convert to complex: `output[..., 0] + 1j * output[..., 1]` → [batch, n_actions, K]

### Step 3.2: QNetwork Class Structure

Follow C51's class structure exactly:

**Class Members**:
- `self.env` - environment reference (for action space)
- `self.n` - number of actions (`env.single_action_space.n`)
- `self.n_frequencies` - K (number of frequency grid points)
- `self.register_buffer("omegas", ...)` - frequency grid (like C51's `atoms`)
- `self.network` - nn.Sequential encoder

**Methods**:
- `get_action(x, action=None)` - returns (action, cf_for_action)
  - Matches C51's `get_action` signature exactly
  - Computes Q-values via collapse, selects argmax action
  - Returns CF for the selected/given action

### Step 3.3: Initialization Considerations

- Use PyTorch default initialization (Xavier/Kaiming via `nn.Linear`)
- No special CF-specific initialization needed initially
- CF will naturally converge to valid values during training

---

## Phase 4: Loss Function Design

### Step 4.1: CF Bellman Target Computation

Compute the target CF using the frequency-domain Bellman equation:

**Target Computation Steps**:
1. Get next-state CFs from target network: Q_target(s', a, ω)
2. Collapse each action's CF to scalar mean
3. Select greedy action: a* = argmax_a E[Q(s', a)]
4. Get target CF for greedy action: Q_target(s', a*, ω)
5. Interpolate at scaled frequencies: Q_target(s', a*, γω)
6. Compute reward CF: exp(iωr)
7. Combine: target_cf = reward_cf × Q_target(s', a*, γω) × (1 - done) + reward_cf × done

**Handling Terminal States**:
- If done=True, future return is 0, so CF of future = 1.0
- Target becomes simply: exp(iωr) (just the immediate reward CF)

### Step 4.2: Loss Computation

**L2 Loss in Frequency Domain**:
- Predicted: Q_online(s, a, ω) for the taken action
- Target: target_cf (detached, no gradient)
- Loss: Mean squared error over ω grid points

**Loss Formula**: L = (1/K) × Σ_k |Q(s,a,ω_k) - target_cf(ω_k)|²

**Note**: For complex tensors, |z|² = Re(z)² + Im(z)²

---

## Phase 5: Training Loop (Matching CleanRL Structure)

The training script must follow cleanrl's single-file, self-contained philosophy with the exact same structure as `c51.py`.

### Step 5.1: Args Dataclass

Use `@dataclass` with tyro for CLI argument parsing, matching C51 exactly:

**Standard Args** (copy from C51):
- `exp_name`, `seed`, `torch_deterministic`, `cuda`
- `track`, `wandb_project_name`, `wandb_entity`
- `capture_video`, `save_model`, `upload_model`, `hf_entity`

**Algorithm-Specific Args** (replace C51's atoms with CF params):
- `env_id`: str = "CartPole-v1"
- `total_timesteps`: int = 500000
- `learning_rate`: float = 2.5e-4
- `num_envs`: int = 1
- `n_frequencies`: int = 128  ← replaces `n_atoms`
- `freq_max`: float = 20.0  ← replaces `v_min`/`v_max`
- `buffer_size`: int = 10000
- `gamma`: float = 0.99
- `target_network_frequency`: int = 500
- `batch_size`: int = 128
- `start_e`, `end_e`, `exploration_fraction`: same as C51
- `learning_starts`: int = 10000
- `train_frequency`: int = 10

### Step 5.2: Main Script Structure

Follow the exact section order from C51:

1. **Imports** (same as C51, add `from cleanrl_utils.cf import ...`)
2. **Args dataclass**
3. **`make_env()` function** (copy from C51)
4. **`QNetwork` class** (CF version)
5. **`linear_schedule()` function** (copy from C51)
6. **`if __name__ == "__main__":` block**

### Step 5.3: Main Block Structure

Follow C51's exact pattern with these sections (use the `# TRY NOT TO MODIFY` comments):

1. Parse args with tyro
2. Set up wandb tracking (if enabled)
3. Set up TensorBoard writer
4. Seeding (random, np, torch)
5. Device setup
6. Environment setup with SyncVectorEnv
7. Network and optimizer initialization
8. ReplayBuffer initialization
9. **Main training loop** (for global_step in range(total_timesteps))
10. Model saving and evaluation (if enabled)
11. Cleanup (envs.close(), writer.close())

### Step 5.4: Action Selection (Inside Main Loop)

Match C51's `get_action` pattern:
1. Compute epsilon via `linear_schedule()`
2. If `random.random() < epsilon`: sample random action
3. Else: `actions, cf = q_network.get_action(torch.Tensor(obs).to(device))`

### Step 5.5: Experience Collection

Copy directly from C51:
- Record episode stats when "final_info" available
- Handle truncation with `real_next_obs`
- Add to replay buffer: `rb.add(obs, real_next_obs, actions, rewards, terminations, infos)`

### Step 5.6: Training Step (ALGO LOGIC Section)

Triggered when `global_step > args.learning_starts` and `global_step % args.train_frequency == 0`:

1. Sample batch: `data = rb.sample(args.batch_size)`
2. **Target computation** (in `torch.no_grad()` block):
   - Get next-state CFs from target network
   - Collapse to Q-values, select greedy actions
   - Compute target CF with Bellman equation
3. **Loss computation**:
   - Get predicted CF for taken actions
   - Compute L2 loss between predicted and target
4. **Logging** (every 100 steps, matching C51):
   - `losses/loss`
   - `losses/q_values` (collapsed mean Q)
   - `charts/SPS`
5. **Optimization**: `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`

### Step 5.7: Target Network Updates

Match C51 exactly (hard copy every N steps):
`target_network.load_state_dict(q_network.state_dict())`

### Step 5.8: Model Saving and Evaluation

Follow C51's pattern:
1. Save model with `.cleanrl_model` extension
2. Run evaluation using `cleanrl_utils/evals/cf_dqn_eval.py`
3. Optionally push to HuggingFace Hub

---

## Phase 6: Hyperparameters

### Step 6.1: CF-Specific Hyperparameters

Based on tabular CVI grid search results:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **W** (frequency range) | 20.0 | Sufficient coverage for typical return distributions |
| **K** (grid points) | 128 | Good balance of accuracy vs compute; tabular achieves 10⁻¹⁵ MAE |
| **Grid Strategy** | three_density_regions | User preference; 50%/30%/20% allocation across 10%/30%/60% of range |
| **Interpolation** | polar | 82.5% of top tabular configs; preserves CF validity |
| **Collapse Method** | gaussian | 96% of excellent configs; phase unwrapping + linear fit |
| **Collapse max_w** | 2.0 | Focus on low-frequency region for mean extraction |

### Step 6.2: Training Hyperparameters (Matching C51 CartPole Defaults)

All hyperparameters match `cleanrl/c51.py` exactly for fair comparison:

| Parameter | Value | Source |
|-----------|-------|--------|
| **total_timesteps** | 500,000 | C51 default |
| **learning_rate** | 2.5e-4 | C51 default |
| **buffer_size** | 10,000 | C51 default |
| **batch_size** | 128 | C51 default |
| **gamma** | 0.99 | C51 default |
| **target_network_frequency** | 500 | C51 default |
| **learning_starts** | 10,000 | C51 default |
| **train_frequency** | 10 | C51 default |
| **start_e** | 1.0 | C51 default |
| **end_e** | 0.05 | C51 default |
| **exploration_fraction** | 0.5 | C51 default |

**Optimizer**: Adam with `eps=0.01 / batch_size` (matching C51)

### Step 6.3: Network Hyperparameters (Matching C51)

| Parameter | Value | Source |
|-----------|-------|--------|
| **Hidden Layer 1** | 120 units | C51 default |
| **Hidden Layer 2** | 84 units | C51 default |
| **Activation** | ReLU | C51 default |
| **Output Size** | n_actions × K × 2 | CF-specific (×2 for real/imag) |

**Note**: C51 uses `n_atoms=101` for its distributional output. Our CF uses `K=128` frequencies, which is comparable in output dimension.

---

## Phase 7: Logging and Monitoring

### Step 7.1: Standard Metrics (Matching CleanRL)

Log these metrics using the exact same keys as C51 for TensorBoard/WandB compatibility:

| Metric Key | When Logged | Description |
|------------|-------------|-------------|
| `charts/episodic_return` | On episode end | Primary performance indicator |
| `charts/episodic_length` | On episode end | Episode length |
| `charts/SPS` | Every 100 steps | Steps per second |
| `losses/loss` | Every 100 steps | Training loss |
| `losses/q_values` | Every 100 steps | Mean collapsed Q-value |

### Step 7.2: CF-Specific Diagnostics (Additional)

Add these CF-specific metrics (logged every 100 steps alongside standard metrics):

| Metric Key | Expected Value | What It Indicates |
|------------|----------------|-------------------|
| `cf/mean_magnitude_at_zero` | ≈ 1.0 | CF at origin should be 1 |
| `cf/max_magnitude` | ≤ 1.0 | CF magnitude bounded by 1 |

**Note**: These are optional diagnostics. If they cause overhead, log less frequently (every 1000 steps).

### Step 7.3: Hyperparameter Logging

At script start, log hyperparameters to TensorBoard (matching C51):
```
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)
```

### Step 7.4: Debugging Visualizations (Optional)

For debugging during development (not in final script):
- Sample CF curves (real/imag vs ω) for a fixed state
- Histogram of collapsed Q-values
- These can be added via wandb image logging if needed

---

## Phase 8: Validation and Testing

### Step 8.1: Sanity Checks Before Training

1. **Forward Pass**: Verify network outputs valid complex tensors with shape [batch, n_actions, K]
2. **Interpolation**: Check that interpolated values lie between grid values; test at γω for known CF
3. **Collapse**: Verify mean extraction on synthetic CFs (e.g., CF of N(μ,σ²) should return μ)
4. **Gradient Flow**: Confirm gradients propagate through all operations with `torch.autograd.grad()`

### Step 8.2: Training Validation

1. **Learning Signal**: Loss should decrease over time
2. **Q-Value Growth**: Collapsed Q-values should increase as policy improves
3. **CF Validity**: |φ(0)| should stay near 1.0 (log this metric)
4. **Episodic Return**: Should trend upward (solve threshold: 195+ avg over 100 episodes)

### Step 8.3: Comparison Baselines

Run all three algorithms on CartPole-v1 with identical non-algorithm hyperparameters:

| Baseline | Script | Command |
|----------|--------|---------|
| DQN | `cleanrl/dqn.py` | `python cleanrl/dqn.py --seed 1` |
| C51 | `cleanrl/c51.py` | `python cleanrl/c51.py --seed 1` |
| CF-DQN | `cleanrl/cf_dqn.py` | `python cleanrl/cf_dqn.py --seed 1` |

Run each with seeds 1, 2, 3 for statistical comparison.

**Success Criteria**: CF-DQN achieves:
- Comparable sample efficiency to C51 (within 20% on timesteps to solve)
- Final performance ≥ 195 average return over 100 episodes
- Stable training (no catastrophic forgetting or divergence)

### Step 8.4: Evaluation Script (`cleanrl_utils/evals/cf_dqn_eval.py`)

Create evaluation script following `c51_eval.py` pattern:

**Function signature**: `evaluate(model_path, make_env, env_id, eval_episodes, run_name, Model, device, epsilon)`

**Implementation**:
1. Load model weights from `.cleanrl_model` file
2. Create evaluation environment
3. Run `eval_episodes` episodes with ε-greedy policy
4. Return list of episodic returns

This enables standardized evaluation and HuggingFace model uploads.

---

## Phase 9: Implementation Order

### Recommended Sequence

1. **Day 1**: Implement `cleanrl_utils/cf.py`
   - `make_omega_grid()` with three_density_regions strategy
   - `interpolate_cf_polar()` with batched operations
   - `collapse_cf_to_mean()` with gaussian method
   - `reward_cf()` for immediate reward CFs
   - Unit tests: verify shapes, compare against numpy reference from cvi_rl

2. **Day 2**: Implement `cleanrl/cf_dqn.py` structure
   - Copy `c51.py` as starting template
   - Update Args dataclass with CF parameters
   - Implement `QNetwork` class with CF output
   - Implement `get_action()` method with collapse
   - Verify network forward pass produces valid shapes

3. **Day 3**: Implement training logic
   - Implement target CF computation in training step
   - Implement L2 loss in frequency domain
   - Wire up the complete training loop
   - Add CF-specific logging metrics
   - Run first training attempt

4. **Day 4**: Debug and validate
   - Fix any numerical issues (NaN, exploding gradients)
   - Verify CF validity metrics during training
   - Tune hyperparameters if needed
   - Achieve stable, improving learning curve

5. **Day 5**: Finalize and document
   - Implement `cleanrl_utils/evals/cf_dqn_eval.py`
   - Run multiple seeds (1, 2, 3) for statistical validity
   - Compare learning curves against C51 and DQN baselines
   - Update documentation with results
   - Tag release and prepare for Atari extension

### File Creation Order

1. `cleanrl_utils/cf.py` — CF operations (no dependencies)
2. `cleanrl/cf_dqn.py` — Main script (depends on cf.py)
3. `cleanrl_utils/evals/cf_dqn_eval.py` — Eval script (depends on cf_dqn.py)

---

## Phase 10: Known Risks and Mitigations

### Risk 1: Numerical Instability in Phase Unwrapping

**Symptom**: NaN or exploding gradients
**Mitigation**: 
- Clamp phase differences before cumsum
- Add small epsilon to denominators
- Use gradient clipping

### Risk 2: CF Validity Violations

**Symptom**: |φ(ω)| > 1 or φ(0) ≠ 1
**Mitigation**:
- Add soft constraint loss: λ × (|φ(0)| - 1)²
- Project magnitudes: min(|φ|, 1.0)
- Monitor and alert if violations occur

### Risk 3: Poor Collapse Estimates

**Symptom**: Q-values noisy or wrong sign
**Mitigation**:
- Start with larger max_w for collapse
- Fall back to LS method if gaussian fails
- Verify on synthetic CFs before training

### Risk 4: Slow Convergence

**Symptom**: Learning curve flat or slow
**Mitigation**:
- Increase K (more frequency resolution)
- Adjust W (broader/narrower range)
- Increase batch size for stable gradients
- Check if target network updates too frequently

---

## Appendix: Quick Reference

### Three Density Regions Grid Configuration

```
Region 1 (Core):    |ω| ≤ 0.1 × W   →  50% of K points
Region 2 (Middle):  0.1W < |ω| ≤ 0.4W  →  30% of K points  
Region 3 (Tails):   0.4W < |ω| ≤ W   →  20% of K points

With W=20.0, K=128:
- Core:   [-2, 2] with ~64 points
- Middle: [-8, -2) ∪ (2, 8] with ~38 points
- Tails:  [-20, -8) ∪ (8, 20] with ~26 points
```

### Key Equations

**CF Bellman Equation**:
```
Q_CF(s, a, ω) = E[exp(iωR) × Q_CF(s', a*, γω) | s, a]
```

**Mean Extraction (Gaussian)**:
```
E[G] = μ where phase(φ(ω)) ≈ μω for small ω
```

**L2 Loss**:
```
L = (1/K) Σ_k |Q_pred(ω_k) - Q_target(ω_k)|²
```

