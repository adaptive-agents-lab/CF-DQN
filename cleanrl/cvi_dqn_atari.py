# CVI-DQN for Atari environments
# Adapted from cvi_dqn.py (CartPole) with Atari-specific wrappers and CNN architecture.
# Hyperparameters match c51_atari.py for fair comparison.

import os
import time
import math
import random
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from collections import deque

from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from cleanrl_utils.buffers import ReplayBuffer

from cleanrl.cvi_utils import polar_interpolation, create_uniform_grid, ifft_collapse_q_values, get_cleaned_target_cf

import torch.profiler


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
    profile: bool = False
    """if toggled, enables the PyTorch profiler (adds overhead, disable for normal training)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    K: int = 128
    """the number of frequency grid points (must be even)"""
    w: float = 1.0
    """the frequency range [-W, W] for the uniform grid"""
    q_min: float = -10.0
    """lower bound of the return distribution support (spatial mask); Atari clips rewards to {-1,0,1}"""
    q_max: float = 10.0
    """upper bound of the return distribution support (spatial mask); Atari clips rewards to {-1,0,1}"""
    buffer_size: int = 500000
    """the replay memory buffer size (500K keeps RAM under 16GB for 84x84x4 uint8 frames)"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """the soft update coefficient for Polyak target network updates"""
    target_network_frequency: int = 10000
    """the frequency at which the target network is hard-updated (0 to disable, use Polyak only)"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training (env steps between gradient updates)"""
    max_grad_norm: float = 10.0
    """the maximum gradient norm for clipping"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk


class CF_QNetwork(nn.Module):
    """
    CF-DQN network for Atari: CNN backbone (identical to C51/DQN) outputting
    characteristic function coefficients instead of atoms or scalar Q-values.
    """
    def __init__(self, envs, actual_grid_size):
        super().__init__()
        self.action_dim = envs.single_action_space.n
        self.K = actual_grid_size
        self.zero_idx = actual_grid_size // 2  # Center of the symmetric grid

        # Same CNN architecture as C51 Atari for fair comparison
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )
        # Output: for each action, K complex coefficients (K * 2 reals)
        self.cf_head = nn.Linear(512, self.action_dim * self.K * 2)

    def forward(self, x):
        # Normalize pixel values to [0, 1] — matches C51/DQN convention
        features = self.network(x / 255.0)
        out = self.cf_head(features)

        out = out.view(out.shape[0], self.action_dim, self.K, 2)
        V_complex = torch.complex(out[..., 0], out[..., 1])

        # Hard normalization to ensure V(0) = 1+0j is always respected.
        # This is mathematically exact: phi(0) = E[e^{i*0*G}] = 1.
        V_at_zero = V_complex[..., self.zero_idx : self.zero_idx + 1]
        self._v_at_zero_mag = torch.abs(V_at_zero).detach()
        V_valid = V_complex / (V_at_zero + 1e-8)

        # Enforce |V(ω)| ≤ 1: a necessary condition for any valid characteristic function.
        magnitude = torch.abs(V_valid)
        self._pre_clamp_max_mag = magnitude.max().detach()
        V_valid = V_valid / torch.clamp(magnitude, min=1.0)

        return V_valid


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    current_time = time.strftime('%Y-%m-%d_%H-%M-%S')
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment (matching C51 baseline)"
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
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")

    writer = SummaryWriter(f"runs/{current_time}")
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

    # env setup — same as C51 Atari
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    #! Init CF-Q-Network and Grid
    recent_returns = deque(maxlen=100)
    omega_grid = create_uniform_grid(K=args.K, W=args.w, device=device)
    actual_grid_size = len(omega_grid)

    q_network = CF_QNetwork(envs, actual_grid_size=actual_grid_size).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
    target_network = CF_QNetwork(envs, actual_grid_size=actual_grid_size).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    episode_count = 0

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    if args.profile:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=100, warmup=10, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"runs/{current_time}/profiler"),
            record_shapes=True,
            profile_memory=True,
            with_stack=False
        )
        prof.start()

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            #! CVI action selection via IFFT
            with torch.no_grad():
                V_complex_all = q_network(torch.Tensor(obs).to(device))
                q_values = ifft_collapse_q_values(omega_grid, V_complex_all, q_min=args.q_min, q_max=args.q_max)
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episode_count += 1
                    ep_return = info['episode']['r']
                    episode_return = ep_return.item() if hasattr(ep_return, "item") else float(ep_return[0] if isinstance(ep_return, (list, np.ndarray)) else ep_return)
                    ep_length = info['episode']['l']
                    episode_length = ep_length.item() if hasattr(ep_length, "item") else int(ep_length[0] if isinstance(ep_length, (list, np.ndarray)) else ep_length)

                    recent_returns.append(episode_return)

                    print(f"global_step={global_step}, episode={episode_count}, episodic_return={episode_return:.2f}, episodic_length={episode_length}")
                    writer.add_scalar("charts/episodic_return", episode_return, global_step)
                    writer.add_scalar("charts/episodic_length", episode_length, global_step)
                    writer.add_scalar("charts/moving_avg_return", np.mean(recent_returns), global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        #! CVI logic — uses train_frequency to match C51 baseline exactly
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)

                with torch.no_grad():
                    #* 1. Get target CFs for all actions
                    target_V_complex_all = target_network(data.next_observations)

                    #* 2. Double DQN: online network SELECTS the greedy next action,
                    #*    target network EVALUATES it. Decouples selection from evaluation,
                    #*    breaking the positive feedback loop that causes Q overestimation.
                    online_V_next_all = q_network(data.next_observations)
                    online_Q_next = ifft_collapse_q_values(omega_grid, online_V_next_all, q_min=args.q_min, q_max=args.q_max)
                    next_actions = torch.argmax(online_Q_next, dim=1)  # selected by online network

                    #* 3. Select the CF of the greedy action (evaluated by target network)
                    batch_idx = torch.arange(args.batch_size, device=device)
                    target_V_next = target_V_complex_all[batch_idx, next_actions]

                    #* 4. Handle terminal states
                    gammas = args.gamma * (1 - data.dones)

                    #* 5. Interpolate at scaled frequencies to account for discounting: V(ω) -> V(γ*ω)
                    interp_V = polar_interpolation(omega_grid, target_V_next, gammas)

                    #* 6. Apply reward rotation: e^{i * w * R} in the frequency domain
                    reward_rotation = torch.exp(1j * omega_grid.view(1, -1) * data.rewards)

                    #* 7. Bellman target, then project onto valid distributions via IFFT cleaning
                    td_target_complex_scalar = reward_rotation * interp_V
                    td_target_complex_scalar = get_cleaned_target_cf(omega_grid, td_target_complex_scalar, q_min=args.q_min, q_max=args.q_max)

                current_Q_complex_all = q_network(data.observations)
                current_V_complex_scalar = current_Q_complex_all[batch_idx, data.actions.flatten()]

                #* Weighted MSE Loss in Frequency Domain with Gaussian Weights
                sigma = 0.3
                weights = torch.exp(-(omega_grid ** 2) / (2 * sigma ** 2))
                weights = weights / weights.sum()
                unweighted_mse = torch.abs(current_V_complex_scalar - td_target_complex_scalar) ** 2

                weighted_mse = torch.sum(weights.view(1, -1) * unweighted_mse, dim=1)
                loss = torch.mean(weighted_mse)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/loss", loss.item(), global_step)

                    current_Q_all = ifft_collapse_q_values(omega_grid, current_Q_complex_all, q_min=args.q_min, q_max=args.q_max)
                    current_Q_taken = current_Q_all[batch_idx, data.actions.flatten()]

                    writer.add_scalar("losses/q_values", current_Q_taken.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                    with torch.no_grad():
                        target_V_diag = target_network(data.observations)
                        target_Q_diag = ifft_collapse_q_values(omega_grid, target_V_diag, q_min=args.q_min, q_max=args.q_max)
                        target_Q_taken_diag = target_Q_diag[batch_idx, data.actions.flatten()]
                        writer.add_scalar("diagnostics/target_q_values", target_Q_taken_diag.mean().item(), global_step)

                    writer.add_scalar("diagnostics/v_at_zero_mag_min", q_network._v_at_zero_mag.min().item(), global_step)
                    writer.add_scalar("diagnostics/pre_clamp_max_magnitude", q_network._pre_clamp_max_mag.item(), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
                optimizer.step()

            # update target network — outside train_frequency gate, matching C51 baseline
            if args.target_network_frequency > 0 and global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())
            elif args.target_network_frequency == 0:
                # Soft Polyak update every env step
                for target_param, param in zip(target_network.parameters(), q_network.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)

        # Inform the profiler that a step has completed
        if args.profile:
            prof.step()

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        model_data = {
            "model_weights": q_network.state_dict(),
            "args": vars(args),
        }
        torch.save(model_data, model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
    if args.profile:
        prof.stop()
