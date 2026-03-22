# CF-DQN Implementation Notes

## Overview

CF-DQN (Characteristic Function Deep Q-Network) is a distributional RL algorithm that represents return distributions in the frequency domain using characteristic functions (CFs). This implementation is based on the CleanRL framework, adapted for CartPole-v1.

## Key Design Choices

### 1. Frequency Grid
- **Strategy**: Three-density regions (`make_omega_grid`)
  - Center (50% points in inner 10%): Dense sampling near ω=0 for accurate mean extraction
  - Middle (30% points in 10-40%): Medium density transition
  - Tails (20% points in outer 60%): Sparse coverage of high frequencies
- **Parameters** (after tuning):
  - `freq_max = 5.0`: Grid spans [-5, 5] (reduced from initial 20.0)
  - `n_frequencies = 128`: Number of frequency points
  - `collapse_max_w = 2.0`: Use frequencies in [-2, 2] for mean extraction

**Rationale**: CartPole Q-values range from 10-500. Initial freq_max=20 was too wide; 5.0 provides better resolution where signal is strong.

### 2. CF Interpolation
- **Method**: Polar interpolation (`interpolate_cf_polar`)
  - Separates magnitude and phase
  - Unwraps phase to handle discontinuities
  - Linear interpolation on both components
  - Reconstructs complex CF
- **Use case**: Bellman operator requires interpolation at γω (scaled frequencies)

**Rationale**: Polar method better preserves CF validity constraints (|φ(ω)| ≤ 1) compared to Cartesian interpolation.

### 3. Mean Extraction
- **Method**: Gaussian collapse (`collapse_cf_to_mean`)
  - Assumes locally Gaussian CF near ω=0
  - Fits phase(ω) ≈ μω via linear regression
  - Returns slope μ as estimated mean
- **Alternative**: Finite difference (diagnostic only)
  - φ'(0) = i·E[G], so E[G] = Im[φ'(0)]
  - Used for validation, not in main algorithm

**Rationale**: Phase slope fitting is robust when |φ(0)| ≈ 1 and provides smooth gradients.

### 4. Network Architecture
- **Same as C51**: 120 → 84 → (n_actions × n_frequencies × 2)
- **Output**: Real and imaginary parts separately
- **No architectural constraints**: φ(0)=1 enforced via loss, not architecture

## Normalization Strategy

### Problem
Valid characteristic functions must satisfy φ(0) = 1 (zeroth moment). Without this constraint:
- CFs don't represent valid probability distributions
- Mean extraction becomes unreliable
- Bellman operator breaks down

### Evolution of Approach

#### ❌ Attempt 1: Hard Normalization in Forward Pass
```python
cf = cf / (cf_at_zero + 1e-8)  # Full complex division by φ(0)
```
**Issue**: Mode collapse - network learned flat CFs (φ(ω) ≈ 1) leading to Q-values ≈ 0.

**Why it failed**: Normalizing both predictions and targets to φ(0)=1 exactly removed gradient signal for learning meaningful CF structure. Trivial constant solution minimized loss.

**Could it work?**: Possibly with modifications:
- **Magnitude-only normalization**: `cf *= 1.0/|φ(0)|` preserves phase derivatives
- **Asymmetric**: Normalize predictions XOR targets, not both
- **Anti-collapse regularization**: Penalize flat CFs (low phase variance)
- **Soft interpolation**: `cf = α*(cf/φ(0)) + (1-α)*cf` for α < 1

#### ❌ Attempt 2: Target-Only Normalization
Normalized targets but not predictions. Similar mode collapse as both converged to flat distributions.

#### ✅ Current Approach: Soft Constraint via Loss Penalty
```python
phi_zero_penalty = mean((|φ_pred(0)| - 1)² + (|φ_target(0)| - 1)²)
loss = cf_mse_loss + 1.0 * phi_zero_penalty
```

**Advantages**:
- Preserves derivative structure (mean information intact)
- Gradual enforcement via penalty weight
- No hard division that could amplify errors
- Applied to both predictions and targets

**Penalty Weight**: Currently 1.0 (10x the initial 0.1). May need further tuning if |φ(0)| remains far from 1.

## WandB Logging Metrics

### Training Metrics
- `losses/loss`: Total loss (CF MSE + φ(0) penalty)
- `losses/phi_zero_penalty`: Soft constraint term
- `losses/q_values`: Mean Q-value from predicted CFs (via collapse_cf_to_mean)
- `losses/q_values_all_mean`: Mean Q-value across all actions
- `losses/q_values_all_max`: Max Q-value across all actions

### CF Diagnostics
- `cf/magnitude_at_zero`: |φ(0)| for predictions (should be ≈1)
- `cf/target_magnitude_at_zero`: |φ(0)| for targets (should be ≈1)
- `cf/max_magnitude`: Max |φ(ω)| across all frequencies (should be ≤1)
- `cf/mean_magnitude`: Average |φ(ω)| across frequencies
- `cf/phase_std`: Standard deviation of phase angles

### Reward Diagnostics
- `debug/raw_rewards_from_env`: Reward immediately after env.step()
- `debug/rewards_before_buffer`: Reward before adding to replay buffer
- `debug/rewards_mean`: Mean reward in sampled batch (should be 1.0 for CartPole)
- `debug/rewards_std`: Std of rewards in batch (should be 0 for CartPole)
- `debug/rewards_min/max`: Min/max rewards in batch
- `debug/reward_cf_phase_std`: Phase variation in reward CF (should be ≈1.31 for r=1)

### Alternative Mean Computation
- `debug/q_manual_mean`: Q-value via finite difference φ'(0) method
  - Compare with `losses/q_values` to validate collapse method
  - Should be positive for CartPole (10-500 range)

### Standard Metrics
- `charts/episodic_return`: Episode return (target: 200-500 for CartPole)
- `charts/episodic_length`: Episode length
- `charts/SPS`: Steps per second

## Issues Encountered and Fixes

### Issue 1: Flat CFs and Zero Q-values
**Symptoms**:
- Phase std dropped from 1.5 to 0.005
- Q-values collapsed to 0.002 (from 3)
- CFs became φ(ω) ≈ 1 for all ω (mode collapse)
- Episodic return stuck at 20

**Cause**: Hard normalization `cf = cf / φ(0)` destroyed gradient information at ω=0

**Fix**: Switched to soft constraint via loss penalty (see Normalization Strategy above)

### Issue 2: Target Normalization Broken
**Symptoms**:
- Target magnitude at zero = 0.42 (should be 1.0)
- Network learned to match invalid targets
- Low loss but poor performance

**Cause**: Interpolation at γω doesn't preserve |φ(0)| = 1

**Fix**: Included targets in soft penalty term (both pred and target penalized)

### Issue 3: Frequency Range Too Wide
**Symptoms**:
- Initial freq_max = 20.0 with collapse_max_w = 2.0
- Only using 10% of frequency grid for mean extraction
- Q-values around 1.5-1.6 (too low)

**Fix**: Reduced freq_max to 5.0, increased collapse_max_w to 2.0
- Better resolution where signal is strong
- More frequencies used for mean extraction

### Issue 3: Max Magnitude > 1
**Symptoms**:
- Max CF magnitude 1.10-1.20 (violates |φ(ω)| ≤ 1)

**Status**: Partially addressed by soft penalty, but not fully resolved
- May need additional magnitude clipping or constraints
- Currently monitoring but not causing major issues

## Remaining Open Issues

### 1. Magnitude Constraint Violations
**Current State**: Max magnitude occasionally exceeds 1.0 (typically 1.05-1.20)

**Impact**: Violates theoretical CF constraints, could affect learning stability

**Potential Solutions**:
- **Magnitude-only normalization**: Scale entire CF so |φ(0)|=1, preserving phase
  ```python
  cf *= 1.0 / (torch.abs(cf_at_zero) + 1e-8)
  ```
  Pro: Enforces constraint without destroying phase gradients
  Con: Doesn't guarantee real-valued φ(0), just |φ(0)|=1
- Add magnitude clipping: `cf = cf / torch.clamp(|cf|, min=1.0)`
- Separate penalty for |φ(ω)| > 1
- Network architecture: Use tanh on magnitude output

**Priority**: Medium (not blocking learning, but theoretically incorrect)

### 2. Performance on CartPole
**Current State**: Episodic return varies (10-20 range initially)

**Target**: 200-500 (CartPole max is 500)

**Status**: Under evaluation with current fixes
- Need to run full training with all fixes applied
- Monitor if soft penalty allows proper learning

### 3. Penalty Weight Tuning
**Current State**: phi_zero_penalty weight = 1.0

**Observation**: Still may need tuning
- If |φ(0)| stays far from 1.0, increase weight
- If mode collapse returns, decrease weight
- Need to balance with CF MSE loss

**Priority**: High (critical for valid CF learning)

### 4. Collapse Method Validation
**Current State**: Two methods available
- `collapse_cf_to_mean` (phase slope fitting)
- Finite difference (diagnostic)

**Question**: Which is more robust?
- Compare q_values vs q_manual_mean
- If diverge significantly, one method is failing
- May need to switch methods or ensemble

**Priority**: Medium (affects Q-value accuracy)

### 5. Interpolation Error Accumulation
**Current State**: Polar interpolation at γω introduces small errors

**Impact**: Targets don't perfectly preserve |φ(0)| = 1
- Soft penalty helps but doesn't eliminate issue
- Errors may accumulate through Bellman iterations

**Potential Solutions**:
- More sophisticated interpolation (spline, higher-order)
- Explicit re-normalization after interpolation
- Use exact CF composition rules when possible

**Priority**: Low (soft penalty mitigates issue)

## Next Steps

1. **Run full training** (500k steps) with all fixes to evaluate:
   - Final episodic return
   - Convergence of phi_zero_penalty
   - Stability of Q-values

2. **Monitor key diagnostics**:
   - `cf/magnitude_at_zero` → should converge to ≈1.0
   - `losses/q_values` → should be positive and increasing
   - `debug/q_manual_mean` → should match q_values

3. **If performance is still poor**:
   - Increase penalty weight to 5.0 or 10.0
   - Consider magnitude-only normalization
   - Try different frequency ranges

## References

- Original CVI paper/implementation in cvi_rl (tabular settings)
- CleanRL C51 implementation (base architecture)

