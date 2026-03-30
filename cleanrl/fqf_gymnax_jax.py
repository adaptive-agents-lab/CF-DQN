"""
Fully Quantile Function (FQF, Yang et al. 2019) — Equinox + vectorised gymnax/Craftax.

Quantile regression loss (IQN-style with τ̂ from fraction network) plus entropy bonus on softmax
proposals. Entropy term is scaled by ``ent_coef`` and ``fqf_lr_factor`` (fraction LR scale).
"""
import os
import time
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from tensorboardX import SummaryWriter

from cleanrl.craftax_env import get_action_dim, get_obs_size, make_env


def huber_loss(u: jnp.ndarray, kappa: float) -> jnp.ndarray:
    abs_u = jnp.abs(u)
    quad = jnp.minimum(abs_u, kappa)
    lin = abs_u - quad
    return 0.5 * quad**2 + kappa * lin


def quantile_huber_loss_iqn(u: jnp.ndarray, tau_i: jnp.ndarray, kappa: float) -> jnp.ndarray:
    indicator = (u < 0).astype(jnp.float32)
    rho = jnp.abs(tau_i[None, :, None] - indicator) * huber_loss(u, kappa) / kappa
    return jnp.mean(rho)


def quantile_huber_loss_fqf(u: jnp.ndarray, tau_hat: jnp.ndarray, kappa: float) -> jnp.ndarray:
    """u: (B, N, Np); tau_hat: (B, N) one τ̂ per quantile row."""
    indicator = (u < 0).astype(jnp.float32)
    rho = jnp.abs(tau_hat[:, :, None] - indicator) * huber_loss(u, kappa) / kappa
    return jnp.mean(rho)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    wandb_tags: str = ""
    save_model: bool = False

    env_id: str = "CartPole-v1"
    total_timesteps: int = 500_000
    learning_rate: float = 2.5e-4
    num_envs: int = 16
    buffer_size: int = 10_000
    gamma: float = 0.99
    tau: float = 0.005
    target_network_frequency: int = 0
    batch_size: int = 128
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 10_000
    utd_ratio: float = 0.1
    max_grad_norm: float = 10.0
    log_interval: int = 10_000
    hidden1: int = 120
    hidden2: int = 84
    hidden3: int = 0
    n_cos: int = 64
    quantile_embedding_dim: int = 64
    num_atoms: int = 32
    """number of quantile proposals N (fraction softmax size)"""
    num_tau_samples: int = 32
    num_tau_prime_samples: int = 32
    num_quantile_samples: int = 32
    kappa: float = 1.0
    ent_coef: float = 0.001
    """entropy coefficient on fraction distribution"""
    fqf_lr_factor: float = 1e-6
    """extra scale on entropy loss (fraction network effective LR)"""


class FQFNetwork(eqx.Module):
    """Encoder + softmax fraction proposals + IQN-style Z(h, τ̂)."""

    layers: list
    fraction_lin: eqx.nn.Linear
    embed_w: eqx.nn.Linear
    head: eqx.nn.Linear
    action_dim: int = eqx.field(static=True)
    n_atoms: int = eqx.field(static=True)
    n_cos: int = eqx.field(static=True)
    quantile_embedding_dim: int = eqx.field(static=True)

    def __init__(
        self,
        obs_size: int,
        action_dim: int,
        n_atoms: int,
        *,
        key,
        hidden1: int,
        hidden2: int,
        hidden3: int,
        n_cos: int,
        quantile_embedding_dim: int,
    ):
        keys = jax.random.split(key, 10)
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.n_cos = n_cos
        self.quantile_embedding_dim = quantile_embedding_dim
        if hidden3 and hidden3 > 0:
            h_out = hidden3
            self.layers = [
                eqx.nn.Linear(obs_size, hidden1, key=keys[0]),
                eqx.nn.Linear(hidden1, hidden2, key=keys[1]),
                eqx.nn.Linear(hidden2, hidden3, key=keys[2]),
            ]
        else:
            h_out = hidden2
            self.layers = [
                eqx.nn.Linear(obs_size, hidden1, key=keys[0]),
                eqx.nn.Linear(hidden1, hidden2, key=keys[1]),
            ]
        self.fraction_lin = eqx.nn.Linear(h_out, n_atoms, key=keys[3])
        self.embed_w = eqx.nn.Linear(n_cos, quantile_embedding_dim, key=keys[4])
        self.head = eqx.nn.Linear(h_out + quantile_embedding_dim, action_dim, key=keys[5])

    def _trunk(self, x):
        for layer in self.layers:
            x = jax.nn.relu(layer(x))
        return x

    def _tau_embed(self, tau: jnp.ndarray) -> jnp.ndarray:
        def one(t: jnp.ndarray) -> jnp.ndarray:
            i = jnp.arange(1, self.n_cos + 1, dtype=jnp.float32)
            cosf = jnp.cos(jnp.pi * i * t)
            return jax.nn.relu(self.embed_w(cosf))

        tau = jnp.atleast_1d(tau)
        return jax.vmap(one)(tau)

    def _z_at(self, h: jnp.ndarray, tau_scalar: jnp.ndarray) -> jnp.ndarray:
        emb = self._tau_embed(jnp.atleast_1d(tau_scalar))[0]
        cat = jnp.concatenate([h, emb], axis=-1)
        return self.head(cat)

    def forward(self, obs: jnp.ndarray):
        """Returns q_mean (A,), z (N, A), tau_hat (N,), probs (N,)"""
        h = self._trunk(obs)
        logits = self.fraction_lin(h)
        probs = jax.nn.softmax(logits)
        tau_knots = jnp.concatenate([jnp.zeros(1, dtype=jnp.float32), jnp.cumsum(probs)])
        tau_hat = 0.5 * (tau_knots[:-1] + tau_knots[1:])
        z = jax.vmap(lambda t: self._z_at(h, t))(tau_hat)
        w = tau_knots[1:] - tau_knots[:-1]
        q_mean = jnp.sum(w[:, None] * z, axis=0)
        return q_mean, z, tau_hat, probs

    def z_sa(self, obs: jnp.ndarray, actions: jnp.ndarray, taus: jnp.ndarray):
        """Z(s, a, τ_i) for batch, taus (K,) — (B, K)"""

        def one(o, a):
            h = self._trunk(o)
            z_all = jax.vmap(lambda t: self._z_at(h, t))(taus)
            return z_all[:, a]

        return jax.vmap(one)(obs, actions)

    def q_values_infer(self, obs: jnp.ndarray, key: jnp.ndarray, num_samples: int) -> jnp.ndarray:
        """Monte Carlo mean over τ ~ U(0,1) like IQN."""
        tau = jax.random.uniform(key, (num_samples,))
        h = self._trunk(obs)
        z = jax.vmap(lambda t: self._z_at(h, t))(tau)
        return jnp.mean(z, axis=0)


class ReplayBufferState(eqx.Module):
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    pos: jnp.ndarray
    size: jnp.ndarray
    capacity: int = eqx.field(static=True)

    @staticmethod
    def create(capacity: int, obs_dim: int):
        return ReplayBufferState(
            obs=jnp.zeros((capacity, obs_dim), dtype=jnp.float32),
            next_obs=jnp.zeros((capacity, obs_dim), dtype=jnp.float32),
            actions=jnp.zeros((capacity,), dtype=jnp.int32),
            rewards=jnp.zeros((capacity,), dtype=jnp.float32),
            dones=jnp.zeros((capacity,), dtype=jnp.float32),
            pos=jnp.array(0, dtype=jnp.int32),
            size=jnp.array(0, dtype=jnp.int32),
            capacity=capacity,
        )


def rb_add(rb, obs, next_obs, action, reward, done):
    idx = rb.pos % rb.capacity
    rb = eqx.tree_at(lambda r: r.obs, rb, rb.obs.at[idx].set(obs))
    rb = eqx.tree_at(lambda r: r.next_obs, rb, rb.next_obs.at[idx].set(next_obs))
    rb = eqx.tree_at(lambda r: r.actions, rb, rb.actions.at[idx].set(action))
    rb = eqx.tree_at(lambda r: r.rewards, rb, rb.rewards.at[idx].set(reward))
    rb = eqx.tree_at(lambda r: r.dones, rb, rb.dones.at[idx].set(done))
    rb = eqx.tree_at(lambda r: r.pos, rb, rb.pos + 1)
    rb = eqx.tree_at(lambda r: r.size, rb, jnp.minimum(rb.size + 1, rb.capacity))
    return rb


def rb_add_batch(rb, obs_batch, next_obs_batch, action_batch, reward_batch, done_batch):
    def add_one(rb, transition):
        obs, next_obs, action, reward, done = transition
        return rb_add(rb, obs, next_obs, action, reward, done), None

    rb, _ = jax.lax.scan(
        add_one, rb, (obs_batch, next_obs_batch, action_batch, reward_batch, done_batch)
    )
    return rb


def rb_sample(rb, key, batch_size):
    indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=rb.size)
    return rb.obs[indices], rb.next_obs[indices], rb.actions[indices], rb.rewards[indices], rb.dones[indices]


class EpisodeStats(eqx.Module):
    episode_returns: jnp.ndarray
    episode_lengths: jnp.ndarray
    returned_episode_returns: jnp.ndarray
    returned_episode_lengths: jnp.ndarray

    @staticmethod
    def create(num_envs: int):
        return EpisodeStats(
            episode_returns=jnp.zeros(num_envs, dtype=jnp.float32),
            episode_lengths=jnp.zeros(num_envs, dtype=jnp.int32),
            returned_episode_returns=jnp.zeros(num_envs, dtype=jnp.float32),
            returned_episode_lengths=jnp.zeros(num_envs, dtype=jnp.int32),
        )


def update_episode_stats(stats, reward, done):
    new_return = stats.episode_returns + reward
    new_length = stats.episode_lengths + 1
    returned_returns = jnp.where(done, new_return, stats.returned_episode_returns)
    returned_lengths = jnp.where(done, new_length, stats.returned_episode_lengths)
    return EpisodeStats(
        episode_returns=jnp.where(done, 0.0, new_return),
        episode_lengths=jnp.where(done, 0, new_length),
        returned_episode_returns=returned_returns,
        returned_episode_lengths=returned_lengths,
    )


def soft_update_target(q_net, target_net, tau):
    q_arrays, _ = eqx.partition(q_net, eqx.is_array)
    t_arrays, t_static = eqx.partition(target_net, eqx.is_array)
    new_arrays = jax.tree.map(lambda p, tp: tau * p + (1 - tau) * tp, q_arrays, t_arrays)
    return eqx.combine(new_arrays, t_static)


def make_train(args: Args):
    env, env_params = make_env(args.env_id)
    obs_size = get_obs_size(env, env_params)
    action_dim = get_action_dim(env, env_params)
    n = args.num_atoms
    k_tau_p = args.num_tau_prime_samples

    num_env_steps = args.total_timesteps // args.num_envs
    chunk_steps = args.log_interval // args.num_envs
    num_chunks = num_env_steps // chunk_steps
    updates_per_step = max(1, round(args.num_envs * args.utd_ratio))

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(learning_rate=args.learning_rate, eps=0.01 / args.batch_size),
    )

    v_reset = jax.vmap(env.reset, in_axes=(0, None))
    v_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def linear_schedule(t):
        slope = (args.end_e - args.start_e) / (args.exploration_fraction * args.total_timesteps)
        return jnp.maximum(slope * t + args.start_e, args.end_e)

    def gradient_step(q_net, target_net, opt_state, rb, key):
        key, sample_key, _tau_key, taup_key = jax.random.split(key, 4)
        s_obs, s_next_obs, s_actions, s_rewards, s_dones = rb_sample(rb, sample_key, args.batch_size)
        bsz = s_obs.shape[0]
        batch_idx = jnp.arange(bsz)
        n_atoms = args.num_atoms

        tau_prime = jax.random.uniform(taup_key, (k_tau_p,))

        def q_mean_next(net, obs):
            def q_one(o):
                qm, _, _, _ = net.forward(o)
                return qm

            return jax.vmap(q_one)(obs)

        def z_sa(net, obs, actions, taus):
            return net.z_sa(obs, actions, taus)

        def loss_fn(net):
            online_q_next = q_mean_next(net, s_next_obs)
            next_actions = jnp.argmax(online_q_next, axis=1)
            fwd = jax.vmap(lambda o: net.forward(o))(s_obs)
            z_b = fwd[1]
            tau_hat_b = fwd[2]
            theta = jax.vmap(lambda zb, a: zb[:, a])(z_b, s_actions)
            theta_t = z_sa(target_net, s_next_obs, next_actions, tau_prime)
            Y = s_rewards[:, None] + args.gamma * (1.0 - s_dones[:, None]) * theta_t
            Y = jax.lax.stop_gradient(Y)
            u = Y[:, None, :] - theta[:, :, None]
            loss_q = quantile_huber_loss_fqf(u, tau_hat_b, args.kappa)

            def ent_one(o):
                _, _, _, pr = net.forward(o)
                return -jnp.sum(pr * jnp.log(pr + 1e-8))

            ent = jax.vmap(ent_one)(s_obs).mean()
            loss = loss_q + args.ent_coef * args.fqf_lr_factor * (-ent)

            qm_b = fwd[0]
            q_mean = qm_b[batch_idx, s_actions].mean()
            q_gap = (qm_b.max(axis=1) - qm_b.min(axis=1)).mean()
            return loss, (q_mean, q_gap)

        (loss, (q_mean, q_gap)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(q_net)
        updates, new_opt_state = optimizer.update(grads, opt_state, eqx.filter(q_net, eqx.is_array))
        new_q_net = eqx.apply_updates(q_net, updates)
        return new_q_net, new_opt_state, key, loss, q_mean, q_gap

    def train_step(runner_state, unused):
        q_net, target_net, opt_state, rb, obs, env_states, ep_stats, key, step_count = runner_state
        global_step = step_count * args.num_envs

        key, action_key, eps_key, infer_key = jax.random.split(key, 4)
        epsilon = linear_schedule(global_step)

        infer_keys = jax.random.split(infer_key, args.num_envs)
        q_values = jax.vmap(lambda o, k: q_net.q_values_infer(o, k, args.num_quantile_samples))(obs, infer_keys)
        greedy_actions = jnp.argmax(q_values, axis=1)
        random_actions = jax.random.randint(action_key, (args.num_envs,), 0, action_dim)
        use_random = jax.random.uniform(eps_key, (args.num_envs,)) < epsilon
        actions = jnp.where(use_random, random_actions, greedy_actions)

        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, args.num_envs)
        next_obs, env_states, rewards, dones, _ = v_step(step_keys, env_states, actions, env_params)
        next_obs = next_obs.reshape(args.num_envs, -1)

        ep_stats = update_episode_stats(ep_stats, rewards, dones)
        rb = rb_add_batch(rb, obs, next_obs, actions, rewards, dones.astype(jnp.float32))

        is_training = global_step > args.learning_starts

        def do_train(operand):
            q_n, target_n, opt_s, rb_, key_ = operand

            def scan_grad(carry, _unused):
                q_n_, opt_s_, key_ = carry
                q_n_, opt_s_, key_, loss, q_mean, q_gap = gradient_step(q_n_, target_n, opt_s_, rb_, key_)
                return (q_n_, opt_s_, key_), (loss, q_mean, q_gap)

            (q_n, opt_s, key_), (losses, q_means, q_gaps) = jax.lax.scan(
                scan_grad, (q_n, opt_s, key_), None, length=updates_per_step
            )

            if args.target_network_frequency > 0:
                should_update = global_step % args.target_network_frequency == 0
                target_n = jax.lax.cond(
                    should_update,
                    lambda pair: pair[0],
                    lambda pair: pair[1],
                    (q_n, target_n),
                )
            else:
                effective_tau = min(args.tau * args.num_envs, 1.0)
                target_n = soft_update_target(q_n, target_n, effective_tau)

            return q_n, target_n, opt_s, key_, losses.mean(), q_means.mean(), q_gaps.mean()

        def skip_train(operand):
            q_n, target_n, opt_s, _rb, key_ = operand
            z = jnp.float32(0.0)
            return q_n, target_n, opt_s, key_, z, z, z

        q_net, target_net, opt_state, key, mean_loss, mean_q, mean_q_gap = jax.lax.cond(
            is_training, do_train, skip_train, (q_net, target_net, opt_state, rb, key)
        )

        new_runner_state = (q_net, target_net, opt_state, rb, next_obs, env_states, ep_stats, key, step_count + 1)
        metrics = {
            "loss": mean_loss,
            "q_values": mean_q,
            "q_gap": mean_q_gap,
            "epsilon": epsilon,
            "returned_episode_returns": ep_stats.returned_episode_returns,
            "returned_episode_lengths": ep_stats.returned_episode_lengths,
            "dones": dones,
            "global_step": global_step,
        }
        return new_runner_state, metrics

    @eqx.filter_jit
    def train_chunk(runner_state):
        return jax.lax.scan(train_step, runner_state, None, length=chunk_steps)

    def init_runner_state(key):
        key, q_key = jax.random.split(key)
        q_net = FQFNetwork(
            obs_size,
            action_dim,
            n,
            key=q_key,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            hidden3=args.hidden3,
            n_cos=args.n_cos,
            quantile_embedding_dim=args.quantile_embedding_dim,
        )
        target_net = FQFNetwork(
            obs_size,
            action_dim,
            n,
            key=q_key,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            hidden3=args.hidden3,
            n_cos=args.n_cos,
            quantile_embedding_dim=args.quantile_embedding_dim,
        )
        opt_state = optimizer.init(eqx.filter(q_net, eqx.is_array))
        rb = ReplayBufferState.create(args.buffer_size, obs_size)
        key, *env_keys = jax.random.split(key, args.num_envs + 1)
        obs, env_states = v_reset(jnp.stack(env_keys), env_params)
        obs = obs.reshape(args.num_envs, -1)
        ep_stats = EpisodeStats.create(args.num_envs)
        return (q_net, target_net, opt_state, rb, obs, env_states, ep_stats, key, jnp.int32(0))

    return train_chunk, init_runner_state, num_chunks, chunk_steps


if __name__ == "__main__":
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        _tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        _wb: dict[str, Any] = dict(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
        if _tags:
            _wb["tags"] = _tags
        wandb.init(**_wb)
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")

    writer = SummaryWriter(f"runs/{current_time}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    key = jax.random.PRNGKey(args.seed)
    train_chunk, init_runner_state, num_chunks, chunk_steps = make_train(args)
    runner_state = init_runner_state(key)

    start_time = time.time()
    episode_count = 0
    recent_returns = []

    for _ in range(num_chunks):
        runner_state, metrics = train_chunk(runner_state)
        dones = jax.device_get(metrics["dones"])
        returns = jax.device_get(metrics["returned_episode_returns"])
        lengths = jax.device_get(metrics["returned_episode_lengths"])
        losses = jax.device_get(metrics["loss"])
        q_vals = jax.device_get(metrics["q_values"])
        q_gaps = jax.device_get(metrics["q_gap"])
        epsilons = jax.device_get(metrics["epsilon"])
        global_steps = jax.device_get(metrics["global_step"])

        for t in range(chunk_steps):
            for e in range(args.num_envs):
                if dones[t, e]:
                    episode_count += 1
                    ep_return = float(returns[t, e])
                    ep_length = int(lengths[t, e])
                    recent_returns.append(ep_return)
                    if len(recent_returns) > 500:
                        recent_returns = recent_returns[-500:]
                    last_logged_step = int(global_steps[t])
                    writer.add_scalar("charts/episodic_return", ep_return, last_logged_step)
                    writer.add_scalar("charts/episodic_length", ep_length, last_logged_step)
                    writer.add_scalar("charts/episodic_return_by_episode", ep_return, episode_count)

        last_step = int(global_steps[-1])
        last_loss = float(losses[-1])
        last_q = float(q_vals[-1])
        last_q_gap = float(q_gaps[-1])
        last_eps = float(epsilons[-1])
        sps = int(last_step / (time.time() - start_time)) if last_step > 0 else 0
        avg_return = float(np.mean(recent_returns)) if recent_returns else 0.0
        print(
            f"step={last_step:>7d} | episodes={episode_count:>5d} | "
            f"avg_return={avg_return:>7.2f} | loss={last_loss:.4f} | q_gap={last_q_gap:.4f} | "
            f"eps={last_eps:.3f} | SPS={sps}"
        )
        writer.add_scalar("charts/moving_avg_return", avg_return, last_step)
        writer.add_scalar("charts/SPS", sps, last_step)
        writer.add_scalar("charts/epsilon", last_eps, last_step)
        writer.add_scalar("losses/loss", last_loss, last_step)
        writer.add_scalar("losses/q_values", last_q, last_step)
        writer.add_scalar("losses/q_gap", last_q_gap, last_step)

    if args.save_model:
        q_net = runner_state[0]
        model_path = f"runs/{run_name}/{args.exp_name}.eqx"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        eqx.tree_serialise_leaves(model_path, q_net)
        print(f"model saved to {model_path}")

    writer.close()
    print(f"Training complete. Total episodes: {episode_count}")
