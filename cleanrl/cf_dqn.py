# CF-DQN: Characteristic Function Deep Q-Network
# Distributional RL in the frequency domain using characteristic functions
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
    wandb_project_name: str = "cleanRL"
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
    
    # CF-DQN specific arguments (replaces C51's n_atoms, v_min, v_max)
    n_frequencies: int = 256
    """number of frequency grid points (K)"""
    freq_max: float = 2.0
    """maximum frequency W (grid spans [-W, W])"""
    collapse_max_w: float = 2.0
    """max |omega| used for gaussian collapse method"""
    
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
    penalty_weight: float = 5.0
    """the weight of the penalty for the φ(0) constraint"""


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


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, n_frequencies=128, freq_max=20.0, collapse_max_w=2.0):
        super().__init__()
        self.env = env
        self.n_frequencies = n_frequencies
        self.freq_max = freq_max
        self.collapse_max_w = collapse_max_w
        self.n = env.single_action_space.n
        
        # Create and register frequency grid as buffer (like C51's atoms)
        # Note: make_omega_grid returns a CPU tensor, we register it as buffer for device handling
        omegas = make_omega_grid(freq_max, n_frequencies)
        self.register_buffer("omegas", omegas)
        
        # Network architecture (matching C51: 120 -> 84 -> output)
        # Output: n_actions * n_frequencies * 2 (real and imaginary parts)
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, self.n * n_frequencies * 2),
        )

    def forward(self, x):
        """
        Forward pass returning CF values for all actions.
        
        Note: φ(0) = 1 is enforced via soft constraint in loss, not hard normalization.
        
        Args:
            x: observations, shape [batch, obs_dim]
        
        Returns:
            cf: complex characteristic functions, shape [batch, n_actions, K]
        """
        batch_size = x.shape[0]
        
        # Get network output: [batch, n_actions * K * 2]
        output = self.network(x)
        
        # Reshape to [batch, n_actions, K, 2] where last dim is [real, imag]
        output = output.view(batch_size, self.n, self.n_frequencies, 2)
        
        # Convert to complex tensor: [batch, n_actions, K]
        cf = torch.complex(output[..., 0], output[..., 1])
        
        return cf

    def get_action(self, x, action=None):
        """
        Get action and corresponding CF (matching C51's interface).
        
        Args:
            x: observations, shape [batch, obs_dim]
            action: optional actions, shape [batch] - if provided, return CF for these actions
        
        Returns:
            action: selected actions, shape [batch]
            cf_for_action: CF values for selected actions, shape [batch, K]
        """
        # Get CF for all actions: [batch, n_actions, K]
        cf_all = self.forward(x)
        
        # Collapse CF to scalar Q-values for action selection
        # q_values: [batch, n_actions]
        q_values = collapse_cf_to_mean(self.omegas, cf_all, max_w=self.collapse_max_w)
        
        if action is None:
            action = torch.argmax(q_values, dim=1)
        
        # Extract CF for the selected/given action: [batch, K]
        batch_indices = torch.arange(len(x), device=x.device)
        cf_for_action = cf_all[batch_indices, action]
        
        return action, cf_for_action
    
    def get_all_cf(self, x):
        """
        Get CF for all actions (used in training for target computation).
        
        Args:
            x: observations, shape [batch, obs_dim]
        
        Returns:
            cf_all: CF values for all actions, shape [batch, n_actions, K]
            q_values: collapsed Q-values, shape [batch, n_actions]
        """
        cf_all = self.forward(x)
        q_values = collapse_cf_to_mean(self.omegas, cf_all, max_w=self.collapse_max_w)
        return cf_all, q_values


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
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
    )
    start_time = time.time()
    episode_count = 0  # Track total number of completed episodes

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, pmf = q_network.get_action(torch.Tensor(obs).to(device))
            actions = actions.cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        
        # DEBUG: Log raw rewards from environment
        if global_step % 100 == 0:
            writer.add_scalar("debug/raw_rewards_from_env", rewards[0] if isinstance(rewards, np.ndarray) else rewards, global_step)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episode_count += 1
                    episode_return = info['episode']['r']
                    episode_length = info['episode']['l']
                    print(f"global_step={global_step}, episode={episode_count}, episodic_return={episode_return}, episodic_length={episode_length}")
                    writer.add_scalar("charts/episodic_return", episode_return, global_step)
                    writer.add_scalar("charts/episodic_length", episode_length, global_step)
                    
                    # Log return by episode count for fair comparison across algorithms
                    writer.add_scalar("charts/episodic_return_by_episode", episode_return, episode_count)
                    
                    # Log return at every 100th episode for milestone tracking
                    if episode_count % 100 == 0:
                        writer.add_scalar("charts/episodic_return_per_100_episodes", episode_return, episode_count)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        
        # DEBUG: Log rewards before adding to buffer
        if global_step % 100 == 0:
            writer.add_scalar("debug/rewards_before_buffer", rewards[0] if isinstance(rewards, np.ndarray) else rewards, global_step)
        
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                
                with torch.no_grad():
                    # Get CF and Q-values for next states from target network
                    next_cf_all, next_q_values = target_network.get_all_cf(data.next_observations)
                    # [batch, n_actions, K], [batch, n_actions]
                    
                    # Select greedy action for next state
                    next_actions = torch.argmax(next_q_values, dim=1)  # [batch]
                    
                    # Get CF for greedy action: [batch, K]
                    batch_indices = torch.arange(args.batch_size, device=device)
                    next_cf_greedy = next_cf_all[batch_indices, next_actions]
                    
                    # Interpolate CF at scaled frequencies (gamma * omega)
                    omegas = target_network.omegas  # [K]
                    scaled_omegas = args.gamma * omegas
                    next_cf_scaled = interpolate_cf_polar(scaled_omegas, omegas, next_cf_greedy)
                    # [batch, K]
                    
                    # Compute reward CF: exp(i * omega * reward)
                    cf_reward = reward_cf(omegas, data.rewards.flatten())  # [batch, K]
                    
                    # CF Bellman target:
                    # target = reward_cf * next_cf_scaled * (1 - done) + reward_cf * done
                    # When done=1, future CF = 1 (zero future return)
                    dones_expanded = data.dones.flatten().unsqueeze(-1)  # [batch, 1]
                    cf_future = next_cf_scaled * (1 - dones_expanded) + (1.0 + 0j) * dones_expanded
                    target_cf = cf_reward * cf_future  # [batch, K]

                # Get predicted CF for taken actions
                _, pred_cf = q_network.get_action(data.observations, data.actions.flatten())
                # [batch, K]
                
                # L2 loss in frequency domain: |pred - target|^2
                loss = complex_mse_loss(pred_cf, target_cf)
                
                # Add soft constraint for φ(0) ≈ 1 (preserves derivative for mean extraction)
                cf_at_zero_idx = torch.argmin(torch.abs(q_network.omegas)).item()
                pred_at_zero = pred_cf[:, cf_at_zero_idx]
                target_at_zero = target_cf[:, cf_at_zero_idx]

                assert pred_at_zero.dim() == 1, f"pred_at_zero should be 1D, got shape {pred_at_zero.shape}"
                assert target_at_zero.dim() == 1, f"target_at_zero should be 1D, got shape {target_at_zero.shape}"
                
                # Penalty for deviating from |φ(0)| = 1
                phi_zero_penalty = torch.mean((torch.abs(pred_at_zero) - 1.0)**2 + 
                                               (torch.abs(target_at_zero) - 1.0)**2)
                loss = loss + args.penalty_weight * phi_zero_penalty

                if global_step % 100 == 0:
                    writer.add_scalar("losses/loss", loss.item(), global_step)
                    writer.add_scalar("losses/phi_zero_penalty", phi_zero_penalty.item(), global_step)
                    
                    # Diagnostic: Check rewards from replay buffer
                    writer.add_scalar("debug/rewards_mean", data.rewards.mean().item(), global_step)
                    writer.add_scalar("debug/rewards_std", data.rewards.std().item(), global_step)
                    writer.add_scalar("debug/rewards_min", data.rewards.min().item(), global_step)
                    writer.add_scalar("debug/rewards_max", data.rewards.max().item(), global_step)
                    
                    # Diagnostic: Check reward CF properties
                    cf_reward_phase_std = torch.angle(cf_reward).std()
                    writer.add_scalar("debug/reward_cf_phase_std", cf_reward_phase_std.item(), global_step)
                    
                    # Log collapsed Q-values for predicted CF
                    q_values_pred = collapse_cf_to_mean(q_network.omegas, pred_cf, max_w=args.collapse_max_w)
                    writer.add_scalar("losses/q_values", q_values_pred.mean().item(), global_step)
                    
                    # Diagnostic: Manually compute mean using finite difference at ω=0
                    zero_idx = torch.argmin(torch.abs(q_network.omegas)).item()
                    if zero_idx > 0 and zero_idx < len(q_network.omegas) - 1:
                        # Central difference: dφ/dω ≈ (φ[i+1] - φ[i-1]) / (ω[i+1] - ω[i-1])
                        dphi_domega = (pred_cf[:, zero_idx+1] - pred_cf[:, zero_idx-1]) / \
                                      (q_network.omegas[zero_idx+1] - q_network.omegas[zero_idx-1])
                        # φ'(0) = i*E[G], so E[G] = Im[φ'(0)] (no negative sign!)
                        manual_mean = dphi_domega.imag
                        writer.add_scalar("debug/q_manual_mean", manual_mean.mean().item(), global_step)
                    
                    # Log Q-values for all actions (using current network)
                    with torch.no_grad():
                        all_cf, all_q_values = q_network.get_all_cf(data.observations)
                        writer.add_scalar("losses/q_values_all_mean", all_q_values.mean().item(), global_step)
                        writer.add_scalar("losses/q_values_all_max", all_q_values.max().item(), global_step)
                    
                    # CF-specific diagnostics - verify normalization
                    cf_mag_at_zero_idx = torch.argmin(torch.abs(q_network.omegas)).item()
                    cf_at_zero = pred_cf[:, cf_mag_at_zero_idx]
                    writer.add_scalar("cf/magnitude_at_zero", torch.abs(cf_at_zero).mean().item(), global_step)
                    writer.add_scalar("cf/max_magnitude", torch.abs(pred_cf).max().item(), global_step)
                    writer.add_scalar("cf/mean_magnitude", torch.abs(pred_cf).mean().item(), global_step)
                    
                    # Log target CF properties
                    target_mag_at_zero = torch.abs(target_cf[:, cf_mag_at_zero_idx]).mean()
                    writer.add_scalar("cf/target_magnitude_at_zero", target_mag_at_zero.item(), global_step)
                    
                    # Log phase characteristics
                    phase_pred = torch.angle(pred_cf)
                    writer.add_scalar("cf/phase_std", phase_pred.std().item(), global_step)
                    
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        model_data = {
            "model_weights": q_network.state_dict(),
            "args": vars(args),
        }
        torch.save(model_data, model_path)
        print(f"model saved to {model_path}")
        
        # TODO: Implement cleanrl_utils/evals/cf_dqn_eval.py for evaluation
        # For now, run a simple evaluation loop
        eval_episodes = 10
        eval_returns = []
        eval_env = gym.make(args.env_id)
        for _ in range(eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            total_reward = 0.0
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
                    if random.random() < args.end_e:
                        action = eval_env.action_space.sample()
                    else:
                        action, _ = q_network.get_action(obs_tensor)
                        action = action.cpu().numpy()[0]
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                total_reward += reward
                done = terminated or truncated
            eval_returns.append(total_reward)
        eval_env.close()
        
        for idx, episodic_return in enumerate(eval_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)
        print(f"Evaluation: mean={np.mean(eval_returns):.2f}, std={np.std(eval_returns):.2f}")

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, eval_returns, repo_id, "CF-DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
