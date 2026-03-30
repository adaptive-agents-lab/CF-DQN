"""
Unified environment factory supporting both gymnax and Craftax environments.

Usage:
    env, env_params = make_env("Breakout-MinAtar")           # gymnax
    env, env_params = make_env("Craftax-Classic-Symbolic-v1") # craftax

The returned (env, env_params) pair exposes the standard interaction loop:
    obs, state = env.reset(rng, env_params)
    obs, state, reward, done, info = env.step(rng, state, action, env_params)
    jax.vmap(env.reset, in_axes=(0, None))
    jax.vmap(env.step,  in_axes=(0, 0, 0, None))
"""
import numpy as np
import jax

_CRAFTAX_PREFIX = "Craftax"


def make_env(env_id: str):
    """Create (env, env_params) for any gymnax or Craftax env_id.

    Supported Craftax env_ids:
        "Craftax-Classic-Symbolic-v1"   obs=1345, actions=17
        "Craftax-Symbolic-v1"           obs=8268, actions=43

    All other ids are forwarded to ``gymnax.make``.
    """
    if env_id.startswith(_CRAFTAX_PREFIX):
        from craftax.craftax_env import make_craftax_env_from_name  # type: ignore
        env = make_craftax_env_from_name(env_id, auto_reset=True)
        env_params = env.default_params
    else:
        import gymnax
        env, env_params = gymnax.make(env_id)
    return env, env_params


def get_obs_size(env, env_params) -> int:
    """Return the flat observation dimension."""
    if hasattr(env, "obs_shape"):
        return int(np.prod(env.obs_shape))
    # Craftax: do one trial reset to infer shape
    rng = jax.random.PRNGKey(0)
    obs, _ = env.reset(rng, env_params)
    return int(np.prod(obs.shape))


def get_action_dim(env, env_params) -> int:
    """Return the number of discrete actions."""
    if hasattr(env, "num_actions"):
        return int(env.num_actions)
    # Craftax
    return int(env.action_space(env_params).n)
