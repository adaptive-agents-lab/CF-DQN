# CVI-DQN: Characteristic Value Iteration Deep Q-Network
# Distributional RL in the frequency domain using characteristic functions
# Based on: "Characteristic Value Iteration" (Farahmand, 2019)
#
# Key design choices:
#   1. Hard φ(0)=1 enforcement in architecture (no penalty term)
#   2. Gradient clipping for stable updates
#   3. No reward scaling — full signal preserved
#   4. freq_max and collapse_max_w are DECOUPLED:
#      - freq_max controls the CF grid range (gradient signal strength)
#      - collapse_max_w controls Q extraction (must satisfy collapse_max_w × Q_max < π)
#   5. Supports both unweighted and frequency-weighted losses
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer
from cleanrl_utils.cf import (
    make_omega_grid,
    interpolate_cf_polar,
    collapse_cf_to_mean,
    reward_cf,
    complex_mse_loss,
    complex_huber_loss,
    weighted_complex_mse_loss,
    weighted_complex_huber_loss,
)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "CVI-DQN"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""

    # CVI-specific arguments
    n_frequencies: int = 128
    """number of frequency grid points (K)"""
    freq_max: float = 2.0
    """maximum frequency W for CF grid (gradient signal strength)"""
    collapse_max_w: float = 2.0
    """max |omega| for Q extraction. Must satisfy collapse_max_w × Q_max < π"""
    sigma_w: float = 0.3
    """Gaussian weighting std for weighted loss types (only used if loss_type=weighted_*)"""
    loss_type: str = "complex_mse"
    """loss function: complex_mse | complex_huber | weighted_mse | weighted_huber"""
    max_grad_norm: float = 10.0
    """gradient clipping norm (0 to disable)"""

    # Target network
    tau: float = 0.005
    """Polyak averaging coefficient (0 = hard update)"""
    use_polyak: bool = True
    """whether to use Polyak averaging instead of hard updates"""

    # Standard DQN arguments
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    reward_norm_clip: float = 3.0  # unused default kept for compatibility


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk


class QNetwork(nn.Module):
    """
    Neural network that outputs characteristic functions φ(s,a,ω) for each action.
    
    Architecture: obs → 120 → 84 → (n_actions × K × 2)
    Output is reshaped to complex tensor [batch, n_actions, K].
    
    φ(0) = 1 is HARD-ENFORCED in the forward pass (not via penalty).
    """
    def __init__(self, env, n_frequencies=128, freq_max=2.0, collapse_max_w=2.0):
        super().__init__()
        self.env = env
        self.n_frequencies = n_frequencies
        self.freq_max = freq_max
        self.collapse_max_w = collapse_max_w
        self.n = env.single_action_space.n

        omegas = make_omega_grid(freq_max, n_frequencies)
        self.register_buffer("omegas", omegas)

        # Find index closest to ω=0 for hard enforcement
        self.zero_idx = torch.argmin(torch.abs(omegas)).item()

        obs_size = int(np.array(env.single_observation_space.shape).prod())
        self.network = nn.Sequential(
            nn.Linear(obs_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, self.n * n_frequencies * 2),
        )

        # Initialize last layer to output ~1+0j (identity CF)
        with torch.no_grad():
            last_layer = self.network[-1]
            last_layer.weight.fill_(0.0)
            bias = torch.zeros_like(last_layer.bias)
            bias[0::2] = 1.0  # Real parts = 1.0
            last_layer.bias.copy_(bias)
            # Small noise to break symmetry
            last_layer.weight.add_(torch.randn_like(last_layer.weight) * 1e-4)
            last_layer.bias.add_(torch.randn_like(last_layer.bias) * 1e-4)

    def forward(self, x):
        batch_size = x.shape[0]
        output = self.network(x.float())
        output = output.view(batch_size, self.n, self.n_frequencies, 2)
        cf = torch.complex(output[..., 0], output[..., 1])

        # HARD ENFORCE φ(0) = 1+0j — eliminates need for penalty term
        cf[:, :, self.zero_idx] = 1.0 + 0j

        return cf

    def get_action(self, x, action=None):
        cf_all = self.forward(x)               # [batch, n_actions, K]
        q_values = collapse_cf_to_mean(self.omegas, cf_all, max_w=self.collapse_max_w)
        if action is None:
            action = torch.argmax(q_values, dim=1)
        batch_indices = torch.arange(len(x), device=x.device)
        cf_for_action = cf_all[batch_indices, action]
        return action, cf_for_action

    def get_all_cf(self, x):
        cf_all = self.forward(x)
        q_values = collapse_cf_to_mean(self.omegas, cf_all, max_w=self.collapse_max_w)
        return cf_all, q_values


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Print operating regime info
    safe_q_max = np.pi / args.freq_max if args.freq_max > 0 else float('inf')
    print(f"\n{'='*60}")
    print(f"CVI-DQN Operating Regime")
    print(f"{'='*60}")
    print(f"  loss_type    = {args.loss_type}")
    print(f"  freq_max     = {args.freq_max}")
    print(f"  Safe Q_max   = {safe_q_max:.1f}  (need freq_max * Q_max < pi)")
    if 'weighted' in args.loss_type:
        print(f"  sigma_w      = {args.sigma_w}")
    print(f"  collapse_max = {args.collapse_max_w}")
    print(f"{'='*60}\n")

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(
        envs,
        n_frequencies=args.n_frequencies,
        freq_max=args.freq_max,
        collapse_max_w=args.collapse_max_w,
    ).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
    target_network = QNetwork(
        envs,
        n_frequencies=args.n_frequencies,
        freq_max=args.freq_max,
        collapse_max_w=args.collapse_max_w,
    ).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
        n_envs=args.num_envs,
    )
    start_time = time.time()
    episode_count = 0

    # Start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(0, args.total_timesteps, args.num_envs):
        # ε-greedy action selection
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _ = q_network.get_action(torch.Tensor(obs).to(device))
            actions = actions.cpu().numpy()

        # Execute action
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Record episodic returns
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episode_count += 1
                    ep_ret = info['episode']['r']
                    ep_len = info['episode']['l']
                    print(f"step={global_step}, ep={episode_count}, return={ep_ret}, len={ep_len}")
                    writer.add_scalar("charts/episodic_return", ep_ret, global_step)
                    writer.add_scalar("charts/episodic_length", ep_len, global_step)
                    writer.add_scalar("charts/episodic_return_by_episode", ep_ret, episode_count)

        # Store transition
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        # Training
        if global_step > args.learning_starts:
            if global_step % args.train_frequency < args.num_envs:
                data = rb.sample(args.batch_size)

                with torch.no_grad():
                    # Target CF computation
                    next_cf_all, next_q_values = target_network.get_all_cf(data.next_observations)
                    next_actions = torch.argmax(next_q_values, dim=1)
                    batch_indices = torch.arange(args.batch_size, device=device)
                    next_cf_greedy = next_cf_all[batch_indices, next_actions]

                    # Interpolate at γω (discount scaling in frequency domain)
                    omegas = target_network.omegas
                    next_cf_scaled = interpolate_cf_polar(args.gamma * omegas, omegas, next_cf_greedy)

                    # Reward CF: exp(iωr)  — RAW rewards, no scaling
                    cf_r = reward_cf(omegas, data.rewards.flatten())

                    # Bellman target: φ_target = φ_r × [φ_future × (1-done) + 1 × done]
                    dones = data.dones.flatten().unsqueeze(-1)
                    cf_future = next_cf_scaled * (1 - dones) + (1.0 + 0j) * dones
                    target_cf = cf_r * cf_future

                # Predicted CF for taken actions
                _, pred_cf = q_network.get_action(data.observations, data.actions.flatten())

                # Loss computation
                if args.loss_type == "complex_mse":
                    loss = complex_mse_loss(pred_cf, target_cf)
                elif args.loss_type == "complex_huber":
                    loss = complex_huber_loss(pred_cf, target_cf)
                elif args.loss_type == "weighted_mse":
                    loss = weighted_complex_mse_loss(pred_cf, target_cf, omegas, sigma_w=args.sigma_w)
                elif args.loss_type == "weighted_huber":
                    loss = weighted_complex_huber_loss(pred_cf, target_cf, omegas, sigma_w=args.sigma_w)
                else:
                    raise ValueError(f"Unknown loss_type: {args.loss_type}")

                # Optimize with gradient clipping
                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
                optimizer.step()

                # Logging (every 100 steps)
                if global_step % 100 == 0:
                    with torch.no_grad():
                        all_cf, all_q_values = q_network.get_all_cf(data.observations)

                    writer.add_scalar("losses/loss", loss.item(), global_step)
                    writer.add_scalar("losses/q_values_mean", all_q_values.mean().item(), global_step)
                    writer.add_scalar("losses/q_values_std", all_q_values.std().item(), global_step)
                    writer.add_scalar("losses/q_values_max", all_q_values.max().item(), global_step)
                    writer.add_scalar("losses/q_values_min", all_q_values.min().item(), global_step)
                    writer.add_scalar("losses/bellman_error", torch.abs(pred_cf - target_cf).mean().item(), global_step)

                    # Phase safety check
                    max_q = abs(all_q_values.max().item())
                    max_phase = args.freq_max * max_q
                    writer.add_scalar("debug/max_phase", max_phase, global_step)
                    writer.add_scalar("debug/phase_safe", float(max_phase < np.pi), global_step)

                    # φ(0) check (should be exactly 1 now)
                    phi0_dev = (torch.abs(pred_cf[:, q_network.zero_idx]) - 1.0).abs().mean().item()
                    writer.add_scalar("debug/phi0_deviation", phi0_dev, global_step)

                    # Gradient norm
                    total_grad_norm = sum(
                        p.grad.data.norm(2).item() ** 2
                        for p in q_network.parameters() if p.grad is not None
                    ) ** 0.5
                    writer.add_scalar("debug/gradient_norm", total_grad_norm, global_step)

                    writer.add_scalar("charts/epsilon", epsilon, global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                    if global_step % 10000 == 0:
                        print(f"\n{'='*60}")
                        print(f"Step {global_step}: Q_mean={all_q_values.mean():.2f}, "
                              f"Q_max={all_q_values.max():.2f}, loss={loss:.6f}, "
                              f"grad={total_grad_norm:.4f}, max_phase={max_phase:.2f}rad")
                        print(f"{'='*60}\n")

            # Target network update
            if global_step % args.target_network_frequency == 0:
                if args.use_polyak:
                    for param, target_param in zip(q_network.parameters(), target_network.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                else:
                    target_network.load_state_dict(q_network.state_dict())

    # Evaluation
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save({"model_weights": q_network.state_dict(), "args": vars(args)}, model_path)
        print(f"model saved to {model_path}")

        eval_episodes = 10
        eval_returns = []
        eval_env = gym.make(args.env_id)
        for _ in range(eval_episodes):
            obs_eval, _ = eval_env.reset()
            done = False
            total_reward = 0.0
            while not done:
                with torch.no_grad():
                    obs_t = torch.as_tensor(np.array(obs_eval), dtype=torch.float32).unsqueeze(0).to(device)
                    action, _ = q_network.get_action(obs_t)
                    action = action.cpu().numpy()[0]
                obs_eval, reward, terminated, truncated, _ = eval_env.step(action)
                total_reward += reward
                done = terminated or truncated
            eval_returns.append(total_reward)
        eval_env.close()
        for i, er in enumerate(eval_returns):
            writer.add_scalar("eval/episodic_return", er, i)
        print(f"Eval: mean={np.mean(eval_returns):.2f}, std={np.std(eval_returns):.2f}")

    envs.close()
    writer.close()
