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

from collections import deque
import numpy as np

from cleanrl_utils.buffers import ReplayBuffer

from cleanrl.cvi_utils import create_three_density_grid, polar_interpolation, gaussian_collapse_q_values


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
    K: int = 64
    """the number of frequency grid points"""
    w: float = 5.0
    """the frequency range [-W, W] for the grid construction during training"""
    w_collapse: float = 2.0
    """the maximum frequency range [-W, W] for the collapse when selecting the greedy action"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """the soft update coefficient for Polyak target network updates"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.00
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.3
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
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
        env.action_space.seed(seed)

        return env

    return thunk


class CF_QNetwork(nn.Module):
    def __init__(self, envs, actual_grid_size):
        super().__init__()
        self.action_dim = envs.single_action_space.n
        self.K = actual_grid_size 
        
        self.network = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
        )
        self.cf_head = nn.Linear(84, self.action_dim * self.K * 2)
        
    def forward(self, x):
        features = self.network(x)
        out = self.cf_head(features)
        
        out = out.view(out.shape[0], self.action_dim, self.K, 2)
        V_complex = torch.complex(out[..., 0], out[..., 1])
        
        # Hard normalization to ensure V(0) = 1+0j is always respected.
        # This provides the inductive bias that outputs are valid CFs,
        # which is critical for gaussian_collapse phase-slope extraction.
        mag_raw = torch.abs(V_complex)
        phase_raw = torch.angle(V_complex)
        
        zero_idx = self.K // 2
        
        mag_at_zero = mag_raw[..., zero_idx : zero_idx + 1]
        phase_at_zero = phase_raw[..., zero_idx : zero_idx + 1]
        
        mag_norm = mag_raw / (mag_at_zero + 1e-8) 
        phase_norm = phase_raw - phase_at_zero
        
        V_valid = mag_norm * torch.complex(torch.cos(phase_norm), torch.sin(phase_norm))
        return V_valid

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
    
    #! Init CF-Q-Network and Grid
    recent_returns = deque(maxlen=500)
    omega_grid = create_three_density_grid(K=args.K, W=args.w, device=device)
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
            #! CVI action selection
            with torch.no_grad():
                V_complex_all = q_network(torch.Tensor(obs).to(device))
                q_values = gaussian_collapse_q_values(omega_grid, V_complex_all, args.w_collapse)
                actions = torch.argmax(q_values, dim=1).cpu().numpy()
            #* C51 action selection for reference
            # actions, pmf = q_network.get_action(torch.Tensor(obs).to(device))
            # actions = actions.cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

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
                    recent_returns.append(episode_return)
                    writer.add_scalar("charts/moving_avg_return", np.mean(recent_returns), global_step)
                    
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
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        #! CVI logic
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                
                with torch.no_grad():
                    # 1. Get target CFs for all actions
                    target_V_complex_all = target_network(data.next_observations)
                    
                    # 2. Collapse to Q-values to find the greedy next action
                    target_Q = gaussian_collapse_q_values(omega_grid, target_V_complex_all, args.w_collapse)
                    next_actions = torch.argmax(target_Q, dim=1)
                    
                    # 3. Select the CF of the greedy action
                    batch_idx = torch.arange(args.batch_size, device=device)
                    target_V_next = target_V_complex_all[batch_idx, next_actions]
                    
                    # 4. Handle terminal states 
                    gammas = args.gamma * (1 - data.dones)
                    
                    # 5. Interpolate at scaled frequencies
                    interp_V = polar_interpolation(omega_grid, target_V_next, gammas)
                    
                    # 6. Apply reward rotation: e^{i * w * R}
                    reward_rotation = torch.exp(1j * omega_grid.view(1, -1) * data.rewards)
                    
                    # 7. Final Bellman Target
                    y_target = reward_rotation * interp_V 

                # --- Online Network Update ---
                current_V_complex_all = q_network(data.observations)
                
                # Select CF for the actions actually taken
                current_V = current_V_complex_all[batch_idx, data.actions.flatten()]
                
                # Compute Complex MSE Loss in Frequency Domain
                loss = torch.mean(torch.abs(current_V - y_target) ** 2)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/loss", loss.item(), global_step)
                    current_Q_all = gaussian_collapse_q_values(omega_grid, current_V_complex_all, args.w_collapse)
                    current_Q_taken = current_Q_all[batch_idx, data.actions.flatten()]
                    writer.add_scalar("losses/q_values", current_Q_taken.mean().item(), global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    # Diagnostics
                    q_gap = (current_Q_all.max(dim=1).values - current_Q_all.min(dim=1).values).mean()
                    writer.add_scalar("diagnostics/q_action_gap", q_gap.item(), global_step)
                    total_norm = sum(p.grad.data.norm(2).item() ** 2 for p in q_network.parameters() if p.grad is not None) ** 0.5
                    writer.add_scalar("diagnostics/grad_norm", total_norm, global_step)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
                optimizer.step()
                
            # Soft Polyak target network update (every training step)
            for target_param, param in zip(target_network.parameters(), q_network.parameters()):
                target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)
                
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        model_data = {
            "model_weights": q_network.state_dict(),
            "args": vars(args),
        }
        torch.save(model_data, model_path)
        print(f"model saved to {model_path}")
        
        
        #! We might need to evaluate in the near future, but for now we can skip!
        # from cleanrl_utils.evals.c51_eval import evaluate
        # episodic_returns = evaluate(
        #     model_path,
        #     make_env,
        #     args.env_id,
        #     eval_episodes=10,
        #     run_name=f"{run_name}-eval",
        #     Model=CF_QNetwork,
        #     device=device,
        #     epsilon=args.end_e,
        # )
        # for idx, episodic_return in enumerate(episodic_returns):
        #     writer.add_scalar("eval/episodic_return", episodic_return, idx)

        # if args.upload_model:
        #     from cleanrl_utils.huggingface import push_to_hub

        #     repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
        #     repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
        #     push_to_hub(args, episodic_returns, repo_id, "CVI", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
