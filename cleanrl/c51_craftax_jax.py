"""
C51 (Distributional DQN) on gymnax or Craftax via cleanrl.craftax_env (Equinox + vectorized envs).

Uses Double DQN-style action selection: greedy action from online expected Q on s', bootstrap distribution from target.
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


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    wandb_tags: str = ""
    save_model: bool = False

    env_id: str = "Craftax-Classic-Symbolic-v1"
    total_timesteps: int = 50_000_000
    learning_rate: float = 1e-4
    num_envs: int = 64
    n_atoms: int = 51
    v_min: float = 0.0
    v_max: float = 22.0
    buffer_size: int = 500_000
    gamma: float = 0.99
    tau: float = 0.005
    target_network_frequency: int = 0
    batch_size: int = 512
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.2
    learning_starts: int = 10_000
    utd_ratio: float = 0.1
    max_grad_norm: float = 10.0
    log_interval: int = 10_000
    hidden1: int = 256
    hidden2: int = 256
    hidden3: int = 128


class C51Network(eqx.Module):
    """Outputs per-action categorical distributions over fixed atoms (softmax over last dim)."""

    layers: list
    last: eqx.nn.Linear
    action_dim: int = eqx.field(static=True)
    n_atoms: int = eqx.field(static=True)

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
    ):
        keys = jax.random.split(key, 6)
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        out_features = action_dim * n_atoms
        if hidden3 and hidden3 > 0:
            self.layers = [
                eqx.nn.Linear(obs_size, hidden1, key=keys[0]),
                eqx.nn.Linear(hidden1, hidden2, key=keys[1]),
                eqx.nn.Linear(hidden2, hidden3, key=keys[2]),
            ]
            self.last = eqx.nn.Linear(hidden3, out_features, key=keys[3])
        else:
            self.layers = [
                eqx.nn.Linear(obs_size, hidden1, key=keys[0]),
                eqx.nn.Linear(hidden1, hidden2, key=keys[1]),
            ]
            self.last = eqx.nn.Linear(hidden2, out_features, key=keys[2])

    def __call__(self, x):
        h = x
        for layer in self.layers:
            h = jax.nn.relu(layer(h))
        logits = self.last(h).reshape(self.action_dim, self.n_atoms)
        return jax.nn.softmax(logits, axis=-1)


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
        add_one,
        rb,
        (obs_batch, next_obs_batch, action_batch, reward_batch, done_batch),
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


def categorical_projection(
    next_pmfs: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
    atoms: jnp.ndarray,
    gamma: float,
    v_min: float,
    v_max: float,
    n_atoms: int,
) -> jnp.ndarray:
    """Project Bellman-updated atoms onto the fixed support (batched)."""
    bsz = next_pmfs.shape[0]
    tz = rewards[:, None] + gamma * (1.0 - dones[:, None]) * atoms[None, :]
    tz = jnp.clip(tz, v_min, v_max)
    delta_z = atoms[1] - atoms[0]
    b = (tz - v_min) / delta_z
    l = jnp.clip(jnp.floor(b), 0, n_atoms - 1).astype(jnp.int32)
    u = jnp.clip(jnp.ceil(b), 0, n_atoms - 1).astype(jnp.int32)
    d_m_l = (u.astype(jnp.float32) + (l == u).astype(jnp.float32) - b) * next_pmfs
    d_m_u = (b - l.astype(jnp.float32)) * next_pmfs
    rows = jnp.arange(bsz)

    def body_atom(j, val):
        val = val.at[rows, l[:, j]].add(d_m_l[:, j])
        val = val.at[rows, u[:, j]].add(d_m_u[:, j])
        return val

    return jax.lax.fori_loop(0, n_atoms, body_atom, jnp.zeros_like(next_pmfs))


def make_train(args: Args):
    env, env_params = make_env(args.env_id)
    obs_size = get_obs_size(env, env_params)
    action_dim = get_action_dim(env, env_params)

    atoms = jnp.asarray(np.linspace(args.v_min, args.v_max, args.n_atoms, dtype=np.float32))

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
        key, sample_key = jax.random.split(key)
        s_obs, s_next_obs, s_actions, s_rewards, s_dones = rb_sample(rb, sample_key, args.batch_size)
        bsz = s_obs.shape[0]
        batch_idx = jnp.arange(bsz)

        online_next = jax.vmap(q_net)(s_next_obs)
        online_q = jnp.sum(online_next * atoms[None, None, :], axis=-1)
        next_actions = jnp.argmax(online_q, axis=1)
        target_next = jax.vmap(target_net)(s_next_obs)
        next_pmfs = target_next[batch_idx, next_actions]

        target_pmfs = categorical_projection(
            next_pmfs,
            s_rewards,
            s_dones,
            atoms,
            args.gamma,
            float(args.v_min),
            float(args.v_max),
            int(args.n_atoms),
        )
        target_pmfs = jax.lax.stop_gradient(target_pmfs)

        def loss_fn(net):
            pmfs = jax.vmap(net)(s_obs)
            old_pmfs = pmfs[batch_idx, s_actions]
            old_pmfs = jnp.clip(old_pmfs, 1e-5, 1.0 - 1e-5)
            loss = -jnp.mean(jnp.sum(target_pmfs * jnp.log(old_pmfs), axis=-1))
            q_all = jnp.sum(pmfs * atoms[None, None, :], axis=-1)
            q_mean = q_all[batch_idx, s_actions].mean()
            q_gap = (q_all.max(axis=1) - q_all.min(axis=1)).mean()
            return loss, (q_mean, q_gap)

        (loss, (q_mean, q_gap)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(q_net)
        updates, new_opt_state = optimizer.update(grads, opt_state, eqx.filter(q_net, eqx.is_array))
        new_q_net = eqx.apply_updates(q_net, updates)
        return new_q_net, new_opt_state, key, loss, q_mean, q_gap

    def train_step(runner_state, unused):
        q_net, target_net, opt_state, rb, obs, env_states, ep_stats, key, step_count = runner_state
        global_step = step_count * args.num_envs

        key, action_key, eps_key = jax.random.split(key, 3)
        epsilon = linear_schedule(global_step)

        pmfs = jax.vmap(q_net)(obs)
        q_vals = jnp.sum(pmfs * atoms[None, None, :], axis=-1)
        greedy_actions = jnp.argmax(q_vals, axis=1)
        random_actions = jax.random.randint(action_key, (args.num_envs,), 0, action_dim)
        use_random = jax.random.uniform(eps_key, (args.num_envs,)) < epsilon
        actions = jnp.where(use_random, random_actions, greedy_actions)

        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, args.num_envs)
        next_obs, env_states, rewards, dones, _infos = v_step(step_keys, env_states, actions, env_params)
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
            is_training,
            do_train,
            skip_train,
            (q_net, target_net, opt_state, rb, key),
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
        q_net = C51Network(
            obs_size,
            action_dim,
            args.n_atoms,
            key=q_key,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            hidden3=args.hidden3,
        )
        target_net = C51Network(
            obs_size,
            action_dim,
            args.n_atoms,
            key=q_key,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            hidden3=args.hidden3,
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

    for _chunk in range(num_chunks):
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
