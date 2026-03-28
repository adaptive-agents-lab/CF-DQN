import jax
import jax.numpy as jnp


def kappa_quadratic(omega):
    """κ(ω) = ω².  Grows quickly at high frequencies, damping the CF magnitude
    so that phase wrapping becomes irrelevant when |φ| → 0."""
    return omega ** 2


def build_analytic_cf(m, sigma, omegas, kappa_fn=kappa_quadratic):
    """Construct the analytic characteristic function from head outputs.

    φ(x, ω) = exp( -½ κ(ω) σ(x,ω)  +  i ω m(x,ω) )

    Args:
        m:       (*, action_dim) — location head output.
        sigma:   (*, action_dim) — spread head output (non-negative).
        omegas:  (*,) or (*, 1) — frequency values (broadcastable to m/sigma).
        kappa_fn: ω → κ(ω), default quadratic.

    Returns:
        Complex array with same shape as m.
    """
    # Ensure omegas broadcasts against (*, action_dim)
    if omegas.ndim < m.ndim:
        omegas = omegas[..., None]  # (*, 1)
    kappa = kappa_fn(omegas)  # (*, 1) or (*, action_dim)
    log_mag = -0.5 * kappa * sigma   # real part of exponent → controls magnitude
    phase = omegas * m                # imaginary part of exponent
    return jnp.exp(log_mag + 1j * phase)


def sample_frequencies(key, num_samples, omega_max):
    """Sample ω uniformly from [-omega_max, omega_max].

    Easy to swap with Gaussian / curriculum-based sampling later.
    """
    
    #TODO: make sure we only sampel form positive frequencies
    return jax.random.uniform(key, shape=(num_samples,), minval=-omega_max, maxval=omega_max)
