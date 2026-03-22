# CF-DQN Algorithm: Training Procedure

## Overview

CF-DQN (Characteristic Function Deep Q-Network) is a **distributional reinforcement learning** algorithm that represents the distribution of returns in the **frequency domain** using characteristic functions (CFs). Unlike C51 which uses discrete probability masses over return bins, CF-DQN learns complex-valued functions φ(ω) that encode distributional information through phase and magnitude.

**Key Insight**: The characteristic function φ(ω) = E[exp(iωG)] is the Fourier transform of the return distribution. The mean return E[G] can be extracted via φ'(0) = iE[G].

---

## Algorithm Components

### 1. Network Architecture

**QNetwork**: Neural network outputting characteristic functions for each action.

**Structure**:
```
Input: State observation (4-dim for CartPole)
  ↓
Layer 1: Linear(obs_dim → 120) + ReLU
  ↓
Layer 2: Linear(120 → 84) + ReLU
  ↓
Layer 3: Linear(84 → n_actions × K × 2)
  ↓
Reshape: [batch, n_actions, K, 2]
  ↓
Convert to Complex: φ(s,a,ω) ∈ ℂ^K  (real + i·imag)
```

**Output Interpretation**:
- `φ(s, a, ω)`: Complex-valued CF at K frequency points for action a in state s
- Shape: `[batch_size, n_actions, K]` where K = 128 frequency points
- **No architectural constraints**: The network can output arbitrary complex values

**Two Networks**:
1. **q_network**: Policy network (updated every training step)
2. **target_network**: Target network (updated every 500 steps)

---

### 2. Frequency Grid Construction

**Frequency Domain**: ω ∈ [-W, W] where W = `freq_max` = 5.0

**Three-Density Grid Strategy** (`make_omega_grid`):

| Region | Range | Points Allocated | Purpose |
|--------|-------|------------------|---------|
| **Center** | \|ω\| ≤ 0.1W | 50% (64 points) | Dense sampling near ω=0 for accurate mean extraction |
| **Middle** | 0.1W < \|ω\| ≤ 0.4W | 30% (38 points) | Medium density transition |
| **Tails** | 0.4W < \|ω\| ≤ W | 20% (26 points) | Sparse coverage of high frequencies |

**Rationale**: 
- Mean extraction requires accurate φ'(0), so we need high resolution near ω=0
- High frequencies (large ω) contribute little to mean but capture tail behavior
- Grid spans [-5, 5] to match CartPole Q-value range (~10-500)

**Key Parameters**:
- `n_frequencies = 128`: Total grid points (K)
- `freq_max = 5.0`: Maximum frequency
- `collapse_max_w = 2.0`: Use frequencies in [-2, 2] for mean extraction

---

### 3. Training Loop

#### **Step 1: Data Collection**

Standard ε-greedy exploration:
```python
if random() < ε:
    action = random_action()
else:
    # Collapse CF to scalar Q-values for action selection
    Q(s, a) = collapse_cf_to_mean(φ(s, a, ω))
    action = argmax_a Q(s, a)
```

Store transitions `(s, a, r, s', done)` in replay buffer.

#### **Step 2: Sample Mini-Batch**

Sample batch of size 128 from replay buffer:
```
batch = {observations, actions, rewards, next_observations, dones}
```

#### **Step 3: Compute Target CF**

This is the **core CF Bellman operator**:

**a) Select Next Action** (Double DQN style):
```python
# Use target network to get CFs for all actions in next state
φ_target(s', a', γω) for all a'  # [batch, n_actions, K]

# Collapse to scalar Q-values
Q_target(s', a') = collapse_cf_to_mean(φ_target(s', a', ω))

# Select greedy action
a* = argmax_a' Q_target(s', a')
```

**b) Interpolate CF at Scaled Frequencies**:
```python
# Get CF for greedy action: φ_target(s', a*, ω)
# Need to evaluate at γω (discount scaling in frequency domain)
φ_future(s', a*, γω) = interpolate_cf_polar(
    target_omegas = γ · ω,
    grid_omegas = ω,
    cf = φ_target(s', a*, ω)
)
```

**Interpolation Method** (`interpolate_cf_polar`):
1. Decompose into polar form: φ = |φ| · exp(i·arg(φ))
2. Unwrap phase to handle 2π discontinuities
3. Linearly interpolate magnitude and unwrapped phase separately
4. Reconstruct: φ_interp = |φ|_interp · exp(i·arg(φ)_interp)

**c) Compute Reward CF**:
```python
# Characteristic function of immediate reward: exp(iωr)
φ_reward(ω) = exp(i · ω · r)  # [batch, K]
```

For CartPole with r=1.0: φ_reward(ω) = exp(iω) = cos(ω) + i·sin(ω)

**d) Apply CF Bellman Operator**:
```python
# Terminal states: future return is 0, so φ_future = 1 (CF of zero)
φ_future = (1 - done) · φ_future(s', a*, γω) + done · 1

# CF composition: φ(r + γG') = φ_r · φ_G' (independence assumption)
φ_target = φ_reward(ω) · φ_future  # [batch, K]
```

**Mathematical Justification**:
- If r and G' are independent: φ_{r+γG'}(ω) = E[exp(iω(r + γG'))] = E[exp(iωr)] · E[exp(iωγG')]
- In practice, they're conditionally independent given (s, a)
- Frequency scaling: φ_{γG'}(ω) = φ_G'(γω)

#### **Step 4: Compute Prediction CF**

```python
# Get CF for the action that was actually taken
φ_pred(s, a, ω) = q_network.get_action(s, a)  # [batch, K]
```

#### **Step 5: Compute Loss**

**a) CF Matching Loss** (L2 in frequency domain):
```python
L_cf = mean(|φ_pred(ω) - φ_target(ω)|²)
     = mean((Re[φ_pred - φ_target])² + (Im[φ_pred - φ_target])²)
```

**b) Normalization Constraint Penalty**:

Valid characteristic functions must satisfy **φ(0) = 1** (zeroth moment).

```python
# Soft constraint: penalize deviation from |φ(0)| = 1
idx_zero = argmin|ω|  # Index of ω closest to 0

L_penalty = mean((|φ_pred(0)| - 1)² + (|φ_target(0)| - 1)²)
```

**Why Soft Penalty Instead of Hard Normalization?**
- Hard normalization `φ ← φ/φ(0)` caused mode collapse (φ(ω) ≈ 1 for all ω)
- Dividing both predictions and targets removed gradient information
- Soft penalty preserves phase derivatives (crucial for mean extraction)
- Applied to both predictions AND targets (targets also violate constraint after interpolation)

**c) Total Loss**:
```python
L_total = L_cf + penalty_weight · L_penalty

where penalty_weight = 5.0
```

#### **Step 6: Gradient Update**

```python
optimizer.zero_grad()
L_total.backward()
optimizer.step()
```

**Optimizer**: Adam with lr=2.5e-4, eps=0.01/batch_size

#### **Step 7: Target Network Update**

Every 500 steps:
```python
target_network.load_state_dict(q_network.state_dict())
```

---

## Mean Extraction: CF → Scalar Q-values

**Method**: Gaussian Collapse (`collapse_cf_to_mean`)

**Theory**: For locally Gaussian CF near ω=0:
```
log φ(ω) ≈ iμω - 0.5σ²ω²

⟹ phase(φ(ω)) ≈ μω

⟹ φ'(0) = iμ  ⟹  μ = Im[φ'(0)]
```

**Implementation** (phase slope fitting):
1. **Select low frequencies**: Use only ω ∈ [-collapse_max_w, collapse_max_w] = [-2, 2]
2. **Compute unwrapped phase**: θ(ω) = arg(φ(ω)) with discontinuities removed
3. **Linear regression**: Fit θ(ω) ≈ μω (line through origin)
   ```
   μ = sum(ω · θ(ω)) / sum(ω²)
   ```
4. **Return mean**: Q(s, a) = μ

**Why This Works**:
- Near ω=0, the CF is smooth and phase is approximately linear
- Slope of phase gives the mean return
- Robust to noise if |φ(0)| ≈ 1

**Alternative Method** (finite difference - diagnostic only):
```python
φ'(0) ≈ (φ(ω₁) - φ(-ω₁)) / (2ω₁)
Q(s, a) = Im[φ'(0)]
```
Used for validation but not in main algorithm.

---

## Hyperparameter Summary

### Core Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_frequencies` | 128 | Number of frequency points K |
| `freq_max` | 5.0 | Maximum frequency (grid: [-5, 5]) |
| `collapse_max_w` | 2.0 | Frequency range for mean extraction |
| `penalty_weight` | 5.0 | Weight for φ(0)=1 constraint |

### Standard DQN Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `learning_rate` | 2.5e-4 | Adam optimizer learning rate |
| `batch_size` | 128 | Mini-batch size |
| `buffer_size` | 10,000 | Replay buffer capacity |
| `gamma` | 0.99 | Discount factor |
| `target_network_frequency` | 500 | Steps between target updates |
| `learning_starts` | 10,000 | Steps before training begins |
| `train_frequency` | 10 | Train every 10 steps |
| `start_e` / `end_e` | 1.0 / 0.05 | ε-greedy exploration |
| `exploration_fraction` | 0.5 | Fraction of training for exploration |

---

## Diagnostic Logging

### Training Metrics
- `losses/loss`: Total loss (CF MSE + penalty)
- `losses/phi_zero_penalty`: Constraint violation term
- `losses/q_values`: Mean Q-value from Gaussian collapse
- `losses/q_values_all_mean/max`: Q-value statistics across actions

### CF Quality Indicators
- `cf/magnitude_at_zero`: |φ(0)| (should be ≈ 1.0)
- `cf/target_magnitude_at_zero`: |φ_target(0)| (should be ≈ 1.0)
- `cf/max_magnitude`: max|φ(ω)| (should be ≤ 1.0, valid CF constraint)
- `cf/mean_magnitude`: Average |φ(ω)| across frequencies
- `cf/phase_std`: Phase variation (higher = more distributional structure)

### Validation Metrics
- `debug/q_manual_mean`: Q-value via finite difference φ'(0)
- `debug/rewards_mean/std/min/max`: Reward statistics from sampled batch
- `debug/reward_cf_phase_std`: Phase variation in reward CF

### Performance Metrics
- `charts/episodic_return`: Episode total reward (target: 200-500 for CartPole)
- `charts/episodic_length`: Episode length
- `charts/SPS`: Training speed (steps per second)

---

## Key Differences from Standard DQN

| Aspect | Standard DQN | CF-DQN |
|--------|-------------|---------|
| **Output** | Scalar Q(s,a) | Complex CF φ(s,a,ω) ∈ ℂ^K |
| **Target** | r + γ max_a' Q(s', a') | φ_r(ω) · φ_future(γω) |
| **Loss** | Huber/MSE on scalars | L2 on complex vectors + φ(0) penalty |
| **Action Selection** | Direct argmax Q | Collapse CF → Q, then argmax |
| **Information** | Point estimate | Full return distribution |

---

## Key Differences from C51

| Aspect | C51 | CF-DQN |
|--------|-----|---------|
| **Domain** | Value space (bins) | Frequency space (ω) |
| **Representation** | Discrete probabilities p_i | Continuous CF φ(ω) |
| **Support** | Fixed [V_min, V_max] | Implicit (encoded in CF) |
| **Projection** | Categorical projection | CF interpolation |
| **Mean** | ∑ p_i · z_i | Collapse via φ'(0) |
| **Constraints** | ∑ p_i = 1, p_i ≥ 0 | \|φ(0)\| = 1, \|φ(ω)\| ≤ 1 |

---

## Theoretical Foundations

### Characteristic Function Properties

1. **Definition**: φ(ω) = E[exp(iωG)] where G is the return
2. **Zeroth Moment**: φ(0) = E[exp(0)] = 1
3. **First Derivative**: φ'(0) = iE[G] (mean return)
4. **Boundedness**: |φ(ω)| ≤ 1 for all ω
5. **Conjugate Symmetry**: If G is real, φ(-ω) = φ̄(ω)

### CF Bellman Operator

For return G = r + γG':
```
φ_G(ω) = E[exp(iω(r + γG'))]
        = E[exp(iωr) · exp(iωγG')]
        = E[exp(iωr)] · E[exp(iωγG')]    [if r ⊥ G' | s,a]
        = φ_r(ω) · φ_G'(γω)
```

**Frequency Scaling**: φ_{γG'}(ω) = E[exp(iωγG')] = φ_G'(γω)

This is the distributional Bellman equation in the frequency domain.

---

## Current Implementation Status

### ✅ Implemented
- Network architecture with complex output
- Three-density frequency grid
- Polar CF interpolation
- Gaussian collapse for mean extraction
- Soft φ(0)=1 constraint
- CF Bellman operator with terminal state handling
- Double DQN style action selection
- Comprehensive diagnostic logging

### ⚠️ Known Issues
1. **Magnitude constraint violations**: max|φ(ω)| occasionally exceeds 1.0 (should be ≤ 1.0)
2. **Normalization not perfect**: |φ(0)| may deviate from 1.0 despite penalty
3. **Performance unknown**: No confirmed successful CartPole training (target: 200-500 episodic return)
4. **Theoretical gap**: Tabular CVI uses full transition distribution P(s', r | s, a); deep version samples single transitions

### 🔧 Potential Improvements
1. **Magnitude-only normalization**: Scale by |φ(0)| without affecting phase
2. **LS collapse method**: Use least-squares quadratic fit instead of phase slope
3. **Tighter frequency range**: Reduce freq_max to 2.5-3.0 for better resolution
4. **Architectural constraints**: Force |φ(ω)| ≤ 1 via sigmoid on magnitude
5. **Empirical validation**: Test on synthetic tabular problems with known ground truth

---

## Training Example

**Command**:
```bash
python cleanrl/cf_dqn.py --env_id CartPole-v1 --total_timesteps 500000 --track
```

**Expected Behavior** (if working correctly):
- Episodic return should increase from ~20 to 200-500 over training
- `cf/magnitude_at_zero` should converge to ≈ 1.0
- `cf/max_magnitude` should stay ≤ 1.05
- `losses/q_values` should be positive and increase
- Phase std should remain non-zero (> 0.5) indicating distributional learning

**Failure Modes**:
- **Mode collapse**: Phase std → 0, Q → 0, agent fails to learn
- **Magnitude explosion**: max|φ(ω)| >> 1, invalid CF
- **Poor collapse**: Q-values negative or diverging from finite difference method
