# Phase Wrapping Limitation in CF-DQN

## Q-Value Extraction Method

CF-DQN represents return distributions as characteristic functions φ(ω), which are complex-valued functions in the frequency domain.

**What is "phase"?**  
A characteristic function φ(ω) = magnitude × e^(i·phase) has two components:
- **Magnitude:** How strong the frequency component is
- **Phase:** The angular rotation in the complex plane = ω × expected_value

The phase encodes the mean of the distribution. As Q-values increase, the phase rotates proportionally.

**Extracting Q-values:**  
Q-values are extracted via the imaginary part of the CF derivative at ω=0:
```
Q = Im[∂φ/∂ω]|_{ω=0}
```

This collapse method works by:
1. Computing the gradient of the phase with respect to frequency
2. Evaluating it at ω=0 (zero frequency = the mean)
3. The imaginary part gives us the expected return (Q-value)

**Why φ(0) = 1 is critical:**  
The characteristic function at zero frequency must equal 1 (φ(0) = 1) because this is the normalization constraint for probability distributions. If φ(0) ≠ 1, the derivative ∂φ/∂ω becomes corrupted and Q-value extraction fails. We enforce this via a soft penalty in the loss function.

This extraction requires the CF gradient at ω=0 to be **smooth and well-defined**. Phase wrapping destroys this smoothness.

## The Instability Problem

The model **reached optimal policy** (episodic returns of 500) but **could not maintain it**. In the high-reward regime (Q-values 300-500), small errors in the CF representation caused large deviations in extracted Q-values, leading to policy collapse and oscillating returns.

## Root Cause: Phase Wrapping

The characteristic function has phase: `phase(ω) = ω × Q_value`

Once phase exceeds π radians, it **wraps around** (like modulo 2π), making gradients meaningless.

**Representational ceiling:**
```
max_representable_Q ≈ π / freq_max
```

For CartPole (optimal return ~500):
- freq_max = 2.0: max_Q ≈ 1.5 → wraps immediately
- freq_max = 0.3: max_Q ≈ 10 → wraps at Q=396 (18.9 rotations)
- freq_max = 0.015: max_Q ≈ 200 → still insufficient
- freq_max = 0.006: max_Q ≈ 500 → barely sufficient, but CF too smooth to learn value distinctions

**The impossible tradeoff:**
- **High freq_max:** Sharp CF for learning distinctions, but phase wraps at low Q-values
- **Low freq_max:** Avoids wrapping at high Q-values, but CF too smooth to represent value differences

## Experimental Evidence

**freq_max = 2.0** (original hyperparameter sweep):
- Reached optimal policy (return = 500) around episode 7000
- Immediately became unstable, oscillating between 500 and <100
- Small CF errors in high-reward regime → large Q-value deviations → policy collapse
- 30-70x slower convergence than PPO/C51

**freq_max = 0.3** (attempting to reduce wrapping):
- Q-values reached ~396 in 100k steps
- Phase: 118 radians = 18.9 rotations
- φ(0) constraint collapsed (deviation 0.87)
- Gradient norm: 0.0118 (vanishing)
- Phase discontinuities: 16% of all frequency points
- Result: Learning stalled at Q≈40

**Reward clipping** (clip_range = 10.0):
- Returns stuck at 10-89, never improving
- Destroyed learning signal by making all trajectories indistinguishable

## Conclusion

CF-DQN is **fundamentally incompatible** with high-return environments:

| Environment | Max Return | Status |
|-------------|------------|---------|
| CartPole-v1 | ~500 | ❌ Impossible |
| Atari | ~10,000 | ❌ Impossible |
| Low-reward tasks | < 50 | ✅ Feasible |

**Use C51, DQN, or PPO for CartPole and similar environments.**
