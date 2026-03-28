import jax
import jax.numpy as jnp


def build_mog_cf(pi, mu, sigma, omegas):
    """Construct the Mixture-of-Gaussians characteristic function.

    φ(ω) = Σ_k π_k · exp(i·ω·μ_k − ½·ω²·σ²_k)

    This CF is **always valid** by construction:
      - φ(0) = Σ π_k = 1  (since π comes from softmax)
      - |φ(ω)| ≤ Σ π_k |exp(...)| ≤ Σ π_k = 1
    And the Bellman operator maps MoG → MoG (closure).

    Args:
        pi:     (..., action_dim, M)  — mixture weights (sum to 1 over M).
        mu:     (..., action_dim, M)  — component means.
        sigma:  (..., action_dim, M)  — component std devs (positive).
        omegas: (N,) — sampled frequencies (positive, exploiting conjugate symmetry).

    Returns:
        Complex array (..., N, action_dim) — CF values at each frequency.
    """
    # Expand dims for broadcasting:
    #   pi/mu/sigma: (..., 1, action_dim, M)  — insert N dimension
    #   omegas:      (N, 1, 1)                — broadcast to (N, action_dim, M)
    pi_e = pi[..., None, :, :]        # (..., 1, action_dim, M)
    mu_e = mu[..., None, :, :]        # (..., 1, action_dim, M)
    sigma_e = sigma[..., None, :, :]  # (..., 1, action_dim, M)
    w = omegas[..., None, None]       # (N, 1, 1)

    phase = w * mu_e                          # (..., N, action_dim, M)
    decay = -0.5 * w ** 2 * sigma_e ** 2      # (..., N, action_dim, M)

    # Per-component CF: exp(decay + i·phase)
    comp_cf = jnp.exp(decay + 1j * phase)     # (..., N, action_dim, M)

    # Weighted sum over mixture components (last axis)
    phi = jnp.sum(pi_e * comp_cf, axis=-1)    # (..., N, action_dim)
    return phi


def mog_q_values(pi, mu):
    """Extract Q-values from MoG parameters.

    Q(s, a) = E[G] = Σ_k π_k · μ_k

    This is a closed-form expression — no IFFT, no grid, no spatial mask.

    Args:
        pi:  (..., action_dim, M) — mixture weights.
        mu:  (..., action_dim, M) — component means.

    Returns:
        (..., action_dim) — Q-values.
    """
    return jnp.sum(pi * mu, axis=-1)


def sample_frequencies(key, num_samples, omega_max, scale=None):
    """Sample ω from a truncated exponential (half-Laplacian) on (0, omega_max].

    Two key design choices:

    1. **Positive-only**: for real-valued returns, φ(-ω) = conj(φ(ω)),
       so negative frequencies are redundant. Sampling only ω > 0
       doubles effective resolution.

    2. **Exponential (half-Laplacian)**: concentrates samples near ω ≈ 0,
       where the CF carries mean/variance information. This replaces
       the explicit Gaussian frequency weighting (σ_w=0.3) from the
       FFT version — the sampling distribution IS the weighting.

    Args:
        key:         JAX PRNG key.
        num_samples: number of frequencies to draw.
        omega_max:   upper bound for frequencies.
        scale:       exponential distribution scale (default: omega_max / 3).

    Returns:
        (num_samples,) array of positive frequencies.
    """
    if scale is None:
        scale = omega_max / 3.0

    # Inverse CDF of truncated exponential on (0, omega_max]:
    #   F(ω) = (1 - exp(-ω/b)) / (1 - exp(-omega_max/b))
    #   ω = -b · ln(1 - u · F(omega_max))
    u = jax.random.uniform(key, shape=(num_samples,))
    max_cdf = 1.0 - jnp.exp(-omega_max / scale)
    omega = -scale * jnp.log(1.0 - u * max_cdf)
    return omega
