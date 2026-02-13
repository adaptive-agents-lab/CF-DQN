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
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer
from cleanrl_utils.cf import (
    make_omega_grid,
    interpolate_cf_polar,
    collapse_cf_to_mean,
    reward_cf,
    complex_mse_loss,
    complex_huber_loss,
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
    reward_scale: float = 1.0  # Scale rewards down
    """scale factor for rewards before CF computation"""
    clip_rewards: bool = False
    """whether to clip rewards to [-1, 1] for CF stability"""
    normalize_rewards: bool = False
    """whether to normalize rewards to zero mean and unit std"""
    reward_clip_range: float = None  # Auto-compute based on freq_max
    """max absolute reward value (auto: π/freq_max to avoid phase wrapping)"""
    tau: float = 0.005
    """polyak averaging coefficient for target network (0 = hard update, 1 = no update)"""
    use_polyak: bool = True
    """whether to use polyak averaging instead of hard updates"""
    
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
    loss_type: str = "complex_mse"
    """the type of loss function to use (complex_mse)"""


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

        # EXPERIMENTAL: Initialize last layer to output ~1+0j (Identity CF)
        # This prevents "random phase wrapping" at initialization which destabilizes bootstrapping
        with torch.no_grad():
            last_layer = self.network[-1]
            last_layer.weight.fill_(0.0) # Zero out weights
            # Set bias: Real parts = 1.0, Imag parts = 0.0
            # Output is [n_actions * n_frequencies * 2]
            # Reshaped later as [n, K, 2] -> last dim is real/imag
            # So indices 0, 2, 4... are Real. 1, 3, 5... are Imag.
            bias = torch.zeros_like(last_layer.bias)
            bias[0::2] = 1.0 # Set every even index (Real) to 1.0
            last_layer.bias.copy_(bias)
            
            # CRITICAL: Add small random noise to break symmetry and allow gradients to flow
            # Without this, Q-values stay identically 0 and agent picks action 0 forever
            noise_scale = 1e-4
            last_layer.weight.add_(torch.randn_like(last_layer.weight) * noise_scale)
            last_layer.bias.add_(torch.randn_like(last_layer.bias) * noise_scale)

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
        n_envs=args.num_envs,
    )
    start_time = time.time()
    episode_count = 0  # Track total number of completed episodes
    

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(0, args.total_timesteps, args.num_envs):
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
        # if global_step % 100 == 0:
        #     writer.add_scalar("debug/raw_rewards_from_env", rewards[0] if isinstance(rewards, np.ndarray) else rewards, global_step)

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
            if global_step % args.train_frequency < args.num_envs:
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
                    
                    #! Clipping or scalaing rewards
                    # Clip rewards to [-1, 1] for CF stability
                    if args.clip_rewards:
                        # Auto-compute safe clipping range: π/ω_max ensures phases < π (no wrapping)
                        clip_range = args.reward_clip_range or (np.pi / args.freq_max)
                        modified_rewards = torch.clamp(data.rewards.flatten(), -clip_range, clip_range)
                        
                    elif args.normalize_rewards:
                        reward_mean = data.rewards.mean()
                        reward_std = data.rewards.std() + 1e-8
                        normalized_rewards = (data.rewards.flatten() - reward_mean) / reward_std
                        # Soft clip to prevent extreme outliers
                        normalized_rewards = torch.clamp(normalized_rewards, -args.reward_norm_clip, args.reward_norm_clip)
                        # Scale to safe range for CF (phases < π)
                        safe_scale = (np.pi / args.freq_max) / args.reward_norm_clip
                        scaled_rewards = normalized_rewards * safe_scale
                    
                    else:
                        modified_rewards = data.rewards.flatten() * args.reward_scale
                    
                    cf_reward = reward_cf(omegas, modified_rewards) 
                    

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
                if args.loss_type == "complex_mse":
                    loss = complex_mse_loss(pred_cf, target_cf)
                if args.loss_type == "complex_huber":
                    loss = complex_huber_loss(pred_cf, target_cf, delta=1.0)

                
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
                    
                    # ==================== COMPREHENSIVE CF DIAGNOSTICS ====================
                    
                    # 0. COMPUTE Q-VALUES FIRST (needed for other diagnostics)
                    with torch.no_grad():
                        all_cf, all_q_values = q_network.get_all_cf(data.observations)
                    
                    # 1. REWARD STATISTICS - Check if normalization/clipping is working
                    writer.add_scalar("debug/rewards_mean", data.rewards.mean().item(), global_step)
                    # writer.add_scalar("debug/rewards_std", data.rewards.std().item(), global_step)
                    # writer.add_scalar("debug/rewards_min", data.rewards.min().item(), global_step)
                    # writer.add_scalar("debug/rewards_max", data.rewards.max().item(), global_step)
                    # writer.add_scalar("debug/modified_rewards_mean", modified_rewards.mean().item(), global_step)
                    # writer.add_scalar("debug/modified_rewards_std", modified_rewards.std().item(), global_step)
                    
                    # 2. PHASE WRAPPING CHECK - The main culprit
                    # CHECK IMMEDIATE REWARD PHASES (should always be safe)
                    max_phase_immediate = (torch.abs(q_network.omegas).max() * torch.abs(modified_rewards).max()).item()
                    # writer.add_scalar("debug/max_phase_immediate_radians", max_phase_immediate, global_step)
                    
                    # CHECK Q-VALUE PHASES (this is where wrapping actually happens!)
                    # Q-values represent cumulative returns - these can be huge
                    max_q_value = all_q_values.max().item()
                    max_phase_qvalues = (torch.abs(q_network.omegas).max() * max_q_value).item()
                    # writer.add_scalar("debug/max_phase_qvalues_radians", max_phase_qvalues, global_step)
                    # writer.add_scalar("debug/max_phase_qvalues_rotations", max_phase_qvalues / (2 * np.pi), global_step)
                    # writer.add_scalar("debug/phase_is_wrapped", float(max_phase_qvalues > np.pi), global_step)
                    
                    # CHECK PREDICTED CF PHASES (actual phase distribution in network output)
                    # pred_cf_phases = torch.angle(pred_cf)
                    # writer.add_scalar("debug/pred_cf_phase_std", pred_cf_phases.std().item(), global_step)
                    # writer.add_scalar("debug/pred_cf_phase_range", 
                    #                 (pred_cf_phases.max() - pred_cf_phases.min()).item(), global_step)
                    
                    # CHECK TARGET CF PHASES (what we're trying to match)
                    # target_cf_phases = torch.angle(target_cf)
                    # writer.add_scalar("debug/target_cf_phase_std", target_cf_phases.std().item(), global_step)
                    # writer.add_scalar("debug/target_cf_phase_range", 
                    #                 (target_cf_phases.max() - target_cf_phases.min()).item(), global_step)
                    
                    # Check actual phase variance in reward CF
                    # cf_reward_phases = torch.angle(cf_reward)
                    # writer.add_scalar("debug/reward_cf_phase_std", cf_reward_phases.std().item(), global_step)
                    # writer.add_scalar("debug/reward_cf_phase_range", 
                    #                 (cf_reward_phases.max() - cf_reward_phases.min()).item(), global_step)
                    
                    # 3. PHI(0) = 1 CONSTRAINT - Check if CF is properly normalized
                    zero_idx = torch.argmin(torch.abs(q_network.omegas)).item()
                    pred_at_zero_mag = torch.abs(pred_at_zero)
                    target_at_zero_mag = torch.abs(target_at_zero)
                    
                    # writer.add_scalar("cf/pred_phi0_magnitude_mean", pred_at_zero_mag.mean().item(), global_step)
                    # writer.add_scalar("cf/pred_phi0_magnitude_std", pred_at_zero_mag.std().item(), global_step)
                    # writer.add_scalar("cf/target_phi0_magnitude_mean", target_at_zero_mag.mean().item(), global_step)
                    # writer.add_scalar("cf/pred_phi0_deviation_from_1", torch.abs(pred_at_zero_mag - 1.0).mean().item(), global_step)
                    # writer.add_scalar("cf/target_phi0_deviation_from_1", torch.abs(target_at_zero_mag - 1.0).mean().item(), global_step)
                    
                    # 4. CF SMOOTHNESS - Check if CF is well-behaved across frequencies
                    # A smooth CF should have low variance in adjacent frequency differences
                    cf_diff = torch.abs(pred_cf[:, 1:] - pred_cf[:, :-1])
                    # writer.add_scalar("cf/smoothness_mean_diff", cf_diff.mean().item(), global_step)
                    # writer.add_scalar("cf/smoothness_max_diff", cf_diff.max().item(), global_step)
                    
                    # Check for discontinuities (sign of phase wrapping issues)
                    phase_pred = torch.angle(pred_cf)
                    phase_diff = torch.abs(phase_pred[:, 1:] - phase_pred[:, :-1])
                    phase_jumps = (phase_diff > np.pi).float().sum().item()
                    # writer.add_scalar("cf/phase_jumps_count", phase_jumps, global_step)
                    # writer.add_scalar("cf/phase_std", phase_pred.std().item(), global_step)
                    
                    # Log phase plot to WandB to visualize the "Sawtooth"
                    fig, ax = plt.subplots(figsize=(10, 5))
                    # Take first element of batch, first action
                    # omegas is [K], phase_pred is [batch, K]
                    # Get omega values (move to cpu)
                    w_cpu = q_network.omegas.detach().cpu().numpy()
                    # Get phases for first sample in batch
                    phi_cpu = phase_pred[0].detach().cpu().numpy()
                    
                    ax.plot(w_cpu, phi_cpu, label="Phase (Predicted)")
                    ax.set_xlabel("Frequency ($\omega$)")
                    ax.set_ylabel("Phase (radians)")
                    ax.set_title(f"Phase Wrapping Visualization (Step {global_step})")
                    ax.set_ylim(-3.5, 3.5) # View range slightly larger than [-pi, pi]
                    ax.axhline(np.pi, color='r', linestyle='--', alpha=0.3)
                    ax.axhline(-np.pi, color='r', linestyle='--', alpha=0.3)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    writer.add_figure("debug/phase_plot", fig, global_step)
                    plt.close(fig)
                    
                    # 5. Q-VALUE STATISTICS - Already computed above
                    writer.add_scalar("losses/q_values_all_mean", all_q_values.mean().item(), global_step)
                    writer.add_scalar("losses/q_values_all_std", all_q_values.std().item(), global_step)
                    writer.add_scalar("losses/q_values_all_max", all_q_values.max().item(), global_step)
                    writer.add_scalar("losses/q_values_all_min", all_q_values.min().item(), global_step)
                    
                    # Q-value consistency: should increase over time
                    q_action_taken = all_q_values[batch_indices, data.actions.flatten()]
                    writer.add_scalar("losses/q_values_taken_action_mean", q_action_taken.mean().item(), global_step)
                    
                    # 6. BELLMAN ERROR - How well does target match prediction
                    bellman_error = torch.abs(pred_cf - target_cf).mean()
                    writer.add_scalar("losses/bellman_error", bellman_error.item(), global_step)
                    
                    # 7. GRADIENT STATISTICS - Check for exploding/vanishing gradients
                    total_grad_norm = 0.0
                    for p in q_network.parameters():
                        if p.grad is not None:
                            total_grad_norm += p.grad.data.norm(2).item() ** 2
                    total_grad_norm = total_grad_norm ** 0.5
                    writer.add_scalar("debug/gradient_norm", total_grad_norm, global_step)
                    
                    # 8. COLLAPSE METHOD VALIDATION - Check if mean extraction is working
                    # Compare finite difference at ω=0 with Gaussian collapse
                    # if zero_idx > 0 and zero_idx < len(q_network.omegas) - 1:
                    #     dphi_domega = (pred_cf[:, zero_idx+1] - pred_cf[:, zero_idx-1]) / \
                    #                   (q_network.omegas[zero_idx+1] - q_network.omegas[zero_idx-1])
                    #     manual_mean = dphi_domega.imag
                        
                    #     q_values_pred = collapse_cf_to_mean(q_network.omegas, pred_cf, max_w=args.collapse_max_w)
                        
                    #     writer.add_scalar("debug/q_manual_mean", manual_mean.mean().item(), global_step)
                    #     writer.add_scalar("debug/q_collapse_mean", q_values_pred.mean().item(), global_step)
                    #     writer.add_scalar("debug/q_method_difference", 
                    #                     torch.abs(manual_mean - q_values_pred).mean().item(), global_step)
                    
                    # 9. TARGET NETWORK DIVERGENCE - Check if target is too different
                    # with torch.no_grad():
                    #     target_cf_all, _ = target_network.get_all_cf(data.observations)
                    #     current_cf_all, _ = q_network.get_all_cf(data.observations)
                    #     network_divergence = torch.abs(target_cf_all - current_cf_all).mean()
                    #     writer.add_scalar("debug/target_network_divergence", network_divergence.item(), global_step)
                    
                    # 10. INTERPOLATION ERROR - Check if gamma scaling is problematic
                    # with torch.no_grad():
                    #     # Compare interpolated CF with direct evaluation at same frequency
                    #     interp_error = torch.abs(next_cf_scaled).std()
                    #     writer.add_scalar("debug/interpolation_complexity", interp_error.item(), global_step)
                    
                    # 11. ACTION DISTRIBUTION - Check if agent is exploring
                    # action_counts = torch.bincount(data.actions.flatten(), minlength=envs.single_action_space.n)
                    # action_probs = action_counts.float() / args.batch_size
                    # action_entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8))
                    # writer.add_scalar("debug/action_entropy", action_entropy.item(), global_step)
                    
                    # 12. CF MAGNITUDE DISTRIBUTION - Should stay bounded
                    # cf_magnitudes = torch.abs(pred_cf)
                    # writer.add_scalar("cf/magnitude_mean", cf_magnitudes.mean().item(), global_step)
                    # writer.add_scalar("cf/magnitude_max", cf_magnitudes.max().item(), global_step)
                    # writer.add_scalar("cf/magnitude_min", cf_magnitudes.min().item(), global_step)
                    # writer.add_scalar("cf/magnitude_at_zero", torch.abs(pred_at_zero).mean().item(), global_step)
                    
                    # Check if CF is collapsing (all values becoming similar)
                    # cf_variance_across_freq = cf_magnitudes.var(dim=1).mean()
                    # writer.add_scalar("cf/variance_across_frequencies", cf_variance_across_freq.item(), global_step)
                    
                    # Target CF properties
                    # target_mag_at_zero = torch.abs(target_at_zero).mean()
                    # writer.add_scalar("cf/target_magnitude_at_zero", target_mag_at_zero.item(), global_step)
                    
                    # DIAGNOSTIC SUMMARY: Print key metrics every 10k steps
                    if global_step % 10000 == 0:
                        print("\n" + "="*80)
                        print(f"DIAGNOSTIC SUMMARY @ step {global_step}")
                        print("="*80)
                        print(f"Phase wrapping: {max_phase_qvalues:.2f} rad ({max_phase_qvalues/(2*np.pi):.2f} rotations)")
                        print(f"  Status: {'⚠️ WRAPPED' if max_phase_qvalues > np.pi else '✓ Safe'}")
                        print(f"φ(0) deviation: pred={torch.abs(pred_at_zero_mag - 1.0).mean():.4f}, "
                              f"target={torch.abs(target_at_zero_mag - 1.0).mean():.4f}")
                        print(f"Q-values: mean={all_q_values.mean():.2f}, std={all_q_values.std():.2f}")
                        print(f"Gradient norm: {total_grad_norm:.4f}")
                        print(f"Bellman error: {bellman_error:.4f}")
                        print(f"Phase jumps: {phase_jumps:.0f} / {pred_cf.shape[0] * (pred_cf.shape[1]-1)}")
                        print("="*80 + "\n")
                    
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                if args.use_polyak:
                    # Soft update: θ_target = τ*θ + (1-τ)*θ_target
                    for param, target_param in zip(q_network.parameters(), target_network.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                else:
                    # Hard update (original)
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
