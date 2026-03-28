#!/usr/bin/env python3
"""
Unit tests for the MoG CF-DQN (no-collapse) implementation.
Run with: python tests/test_cvi_nocollapse_jax.py
"""
import sys
import numpy as np


def test_sample_frequencies_positive_only():
    """Frequencies should be positive (conjugate symmetry exploitation)."""
    import jax
    from cleanrl.cvi_utils_nocollapse_jax import sample_frequencies

    key = jax.random.PRNGKey(42)
    for N in [8, 32, 64]:
        for omega_max in [0.5, 1.0, 2.0]:
            omegas = sample_frequencies(key, N, omega_max)
            assert omegas.shape == (N,), f"Wrong shape: {omegas.shape}"
            assert float(omegas.min()) > 0, f"Got non-positive ω: {float(omegas.min())}"
            assert float(omegas.max()) <= omega_max, f"Above omega_max"
    print("✓ sample_frequencies (positive-only, half-Laplacian): PASS")


def test_sample_frequencies_concentration():
    """Half-Laplacian should concentrate samples near ω=0."""
    import jax
    import jax.numpy as jnp
    from cleanrl.cvi_utils_nocollapse_jax import sample_frequencies

    key = jax.random.PRNGKey(0)
    omegas = sample_frequencies(key, 10000, omega_max=1.0)
    # With scale=1/3, most samples should be near 0
    fraction_below_half = float(jnp.mean(omegas < 0.5))
    assert fraction_below_half > 0.6, (
        f"Expected >60% of samples below 0.5, got {fraction_below_half:.1%}")
    print("✓ sample_frequencies concentration near ω=0: PASS")


def test_build_mog_cf_at_zero():
    """At ω=0, MoG CF must equal 1+0j (Σ π_k · exp(0) = Σ π_k = 1)."""
    import jax.numpy as jnp
    from cleanrl.cvi_utils_nocollapse_jax import build_mog_cf

    action_dim, M = 2, 4
    pi = jnp.ones((action_dim, M)) / M  # uniform weights
    mu = jnp.array([[1.0, 2.0, -1.0, 5.0], [0.5, -3.0, 2.0, 0.0]])
    sigma = jnp.array([[0.5, 1.0, 0.3, 2.0], [1.5, 0.2, 0.8, 1.0]])

    # Use a very small positive ω (not exactly 0 since we sample positive only)
    omegas = jnp.array([1e-8])
    phi = build_mog_cf(pi, mu, sigma, omegas)  # (1, action_dim)

    np.testing.assert_allclose(np.array(phi.real), 1.0, atol=1e-5,
        err_msg="Re(φ(0)) should be 1")
    np.testing.assert_allclose(np.array(phi.imag), 0.0, atol=1e-4,
        err_msg="Im(φ(0)) should be 0")
    print("✓ build_mog_cf at ω≈0: PASS")


def test_build_mog_cf_magnitude_bound():
    """|φ(ω)| ≤ 1 for any valid MoG parameters."""
    import jax
    import jax.numpy as jnp
    from cleanrl.cvi_utils_nocollapse_jax import build_mog_cf

    key = jax.random.PRNGKey(0)
    A, M = 4, 8
    pi = jax.nn.softmax(jax.random.normal(key, (A, M)), axis=-1)
    mu = jax.random.normal(jax.random.PRNGKey(1), (A, M)) * 10
    sigma = jax.nn.softplus(jax.random.normal(jax.random.PRNGKey(2), (A, M)))
    omegas = jnp.linspace(0.01, 5.0, 200)

    phi = build_mog_cf(pi, mu, sigma, omegas)  # (200, 4)
    magnitudes = np.array(jnp.abs(phi))
    assert np.all(magnitudes <= 1.0 + 1e-5), (
        f"|φ| > 1 detected: max={magnitudes.max():.6f}")
    print("✓ build_mog_cf |φ| ≤ 1: PASS")


def test_mog_q_values():
    """Q = Σ π_k μ_k should match manual computation."""
    import jax.numpy as jnp
    from cleanrl.cvi_utils_nocollapse_jax import mog_q_values

    pi = jnp.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]])  # (2 actions, 3 components)
    mu = jnp.array([[10.0, 20.0, 30.0], [5.0, 15.0, 25.0]])

    q = mog_q_values(pi, mu)
    expected = jnp.array([
        0.5 * 10 + 0.3 * 20 + 0.2 * 30,  # = 17.0
        0.1 * 5 + 0.8 * 15 + 0.1 * 25,   # = 15.0
    ])
    np.testing.assert_allclose(np.array(q), np.array(expected), atol=1e-5,
        err_msg="Q-values don't match Σ π_k μ_k")
    print("✓ mog_q_values: PASS")


def test_bellman_closure():
    """Bellman backup of MoG produces valid CF (still MoG by construction)."""
    import jax
    import jax.numpy as jnp
    from cleanrl.cvi_utils_nocollapse_jax import build_mog_cf

    M = 4
    pi = jax.nn.softmax(jnp.ones(M))
    mu = jnp.array([5.0, 10.0, 15.0, 20.0])
    sigma = jnp.array([1.0, 2.0, 0.5, 1.5])

    reward = 1.0
    gamma = 0.99

    # Bellman-shifted MoG
    bellman_mu = reward + gamma * mu
    bellman_sigma = gamma * sigma
    bellman_pi = pi

    omegas = jnp.linspace(0.01, 3.0, 100)

    # Build CF from Bellman-shifted params
    phi_bellman = build_mog_cf(
        bellman_pi[None, :],    # (1, M) — 1 action
        bellman_mu[None, :],    # (1, M)
        bellman_sigma[None, :], # (1, M)
        omegas,                 # (100,)
    )  # (100, 1)

    # Verify validity
    mags = np.array(jnp.abs(phi_bellman))
    assert np.all(mags <= 1.0 + 1e-5), f"Bellman CF |φ| > 1: {mags.max()}"

    # Compare with manual: e^{iωr} · φ(s', γω)
    phi_original = build_mog_cf(pi[None, :], mu[None, :], sigma[None, :], gamma * omegas)
    reward_rotation = jnp.exp(1j * omegas[:, None] * reward)
    phi_manual = reward_rotation * phi_original

    np.testing.assert_allclose(
        np.array(phi_bellman), np.array(phi_manual), atol=1e-5,
        err_msg="Bellman closure: analytic ≠ manual e^{iωr}·φ(γω)")
    print("✓ Bellman closure: PASS")


def test_network_forward_shapes():
    """Test that CF_QNetwork produces correct output shapes."""
    import jax
    import jax.numpy as jnp
    from cleanrl.cvi_dqn_nocollapse_jax import CF_QNetwork

    obs_size, action_dim, M = 4, 2, 8
    key = jax.random.PRNGKey(0)
    net = CF_QNetwork(obs_size, action_dim, M, key=key)

    obs = jnp.ones(obs_size)
    pi, mu, sigma = net(obs)

    assert pi.shape == (action_dim, M), f"pi shape wrong: {pi.shape}"
    assert mu.shape == (action_dim, M), f"mu shape wrong: {mu.shape}"
    assert sigma.shape == (action_dim, M), f"sigma shape wrong: {sigma.shape}"

    # π sums to 1 per action
    np.testing.assert_allclose(np.array(pi.sum(axis=-1)), 1.0, atol=1e-5,
        err_msg="π doesn't sum to 1")
    # σ is positive
    assert np.all(np.array(sigma) > 0), "σ must be positive"
    print("✓ CF_QNetwork forward shapes: PASS")


def test_network_q_values():
    """Test that q_values() returns correct shape and is deterministic."""
    import jax
    import jax.numpy as jnp
    from cleanrl.cvi_dqn_nocollapse_jax import CF_QNetwork

    obs_size, action_dim, M = 4, 2, 8
    key = jax.random.PRNGKey(0)
    net = CF_QNetwork(obs_size, action_dim, M, key=key)

    obs = jnp.ones(obs_size)
    q = net.q_values(obs)
    assert q.shape == (action_dim,), f"Q shape wrong: {q.shape}"
    assert q.dtype == jnp.float32, f"Q dtype wrong: {q.dtype}"

    # Should be deterministic
    q2 = net.q_values(obs)
    np.testing.assert_array_equal(np.array(q), np.array(q2),
        err_msg="q_values not deterministic")

    # Q = Σ π_k μ_k should match
    pi, mu, _ = net(obs)
    q_manual = np.array(jnp.sum(pi * mu, axis=-1))
    np.testing.assert_allclose(np.array(q), q_manual, atol=1e-6,
        err_msg="q_values ≠ Σ π_k μ_k")
    print("✓ CF_QNetwork q_values: PASS")


def test_network_cf_valid():
    """Test that the MoG CF from network outputs satisfies validity conditions."""
    import jax
    import jax.numpy as jnp
    from cleanrl.cvi_dqn_nocollapse_jax import CF_QNetwork
    from cleanrl.cvi_utils_nocollapse_jax import build_mog_cf

    obs_size, action_dim, M = 4, 2, 8
    key = jax.random.PRNGKey(0)
    net = CF_QNetwork(obs_size, action_dim, M, key=key)

    obs = jnp.ones(obs_size)
    pi, mu, sigma = net(obs)

    N = 64
    omegas = jnp.linspace(0.01, 3.0, N)
    phi = build_mog_cf(pi, mu, sigma, omegas)  # (N, action_dim)

    # Shape check
    assert phi.shape == (N, action_dim), f"φ shape wrong: {phi.shape}"

    # |φ(ω)| ≤ 1
    mags = np.array(jnp.abs(phi))
    assert np.all(mags <= 1.0 + 1e-5), f"|φ| > 1: max={mags.max()}"

    # φ(ω→0) ≈ 1+0j
    tiny_omega = jnp.array([1e-8])
    phi_zero = build_mog_cf(pi, mu, sigma, tiny_omega)  # (1, action_dim)
    np.testing.assert_allclose(np.array(phi_zero.real), 1.0, atol=1e-4)
    np.testing.assert_allclose(np.array(phi_zero.imag), 0.0, atol=1e-4)

    print("✓ Network CF validity: PASS")


def test_network_batched_vmap():
    """Test that vmap over batch of states works."""
    import jax
    import jax.numpy as jnp
    from cleanrl.cvi_dqn_nocollapse_jax import CF_QNetwork
    from cleanrl.cvi_utils_nocollapse_jax import build_mog_cf

    obs_size, action_dim, M = 4, 2, 8
    key = jax.random.PRNGKey(0)
    net = CF_QNetwork(obs_size, action_dim, M, key=key)

    batch = 8
    obs_batch = jnp.ones((batch, obs_size))

    # Batched forward
    pi, mu, sigma = jax.vmap(net)(obs_batch)
    assert pi.shape == (batch, action_dim, M), f"Batched pi shape wrong: {pi.shape}"
    assert mu.shape == (batch, action_dim, M), f"Batched mu shape wrong: {mu.shape}"
    assert sigma.shape == (batch, action_dim, M), f"Batched sigma shape wrong: {sigma.shape}"

    # Batched q_values
    q_batch = jax.vmap(net.q_values)(obs_batch)
    assert q_batch.shape == (batch, action_dim), f"Batched Q shape wrong: {q_batch.shape}"

    print("✓ Batched vmap: PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("MoG CF-DQN (No-Collapse) Verification")
    print("=" * 60)

    tests = [
        test_sample_frequencies_positive_only,
        test_sample_frequencies_concentration,
        test_build_mog_cf_at_zero,
        test_build_mog_cf_magnitude_bound,
        test_mog_q_values,
        test_bellman_closure,
        test_network_forward_shapes,
        test_network_q_values,
        test_network_cf_valid,
        test_network_batched_vmap,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: FAIL - {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed!")
