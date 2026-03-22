import os
import time
import math
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
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    K: int = 128
    """the number of frequency grid points (must be even)"""
    w: float = 1.0
    """the frequency range [-W, W] for the uniform grid"""
    q_min: float = 0.0
    """lower bound of the return distribution support (spatial mask); 0 for non-negative reward envs"""
    q_max: float = 100.0
    """upper bound of the return distribution support (spatial mask); for CartPole gamma=0.99: R_max/(1-gamma)=100"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """the soft update coefficient for Polyak target network updates"""
    target_network_frequency: int = 1000
    """the frequency at which the target network is hard-updated (0 to disable, use Polyak only)"""
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
    utd_ratio: float = 0.1
    """the update-to-data ratio (gradient steps per env step). E.g., 1.0 = train every step, 0.1 = train every 10 steps, 2.0 = 2 updates per step. Default 0.1 matches train_frequency=10."""
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
        self.zero_idx = actual_grid_size // 2  # Center of the symmetric grid
        
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
        # This is mathematically exact: phi(0) = E[e^{i*0*G}] = 1.
        V_at_zero = V_complex[..., self.zero_idx : self.zero_idx + 1]
        self._v_at_zero_mag = torch.abs(V_at_zero).detach()  # Diagnostic: track raw |V(0)|
        V_valid = V_complex / (V_at_zero + 1e-8)
        
        # Enforce |V(ω)| ≤ 1: a necessary condition for any valid characteristic function.
        # Scale down where |V| > 1 while preserving the phase.
        magnitude = torch.abs(V_valid)
        self._pre_clamp_max_mag = magnitude.max().detach()  # Diagnostic: how much clipping is needed
        V_valid = V_valid / torch.clamp(magnitude, min=1.0)
        
        return V_valid

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    current_time = time.strftime('%Y-%m-%d_%H-%M-%S')
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

    # env setup
    envs = gym.vector.AsyncVectorEnv( #TODO: to change to AsyncVectorEnv to enable parallel envs
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    
    #! Init CF-Q-Network and Grid
    recent_returns = deque(maxlen=500)
    omega_grid = create_uniform_grid(K=args.K, W=args.w, device=device)
    actual_grid_size = len(omega_grid)

    q_network = CF_QNetwork(envs, actual_grid_size=actual_grid_size).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
    target_network = CF_QNetwork(envs, actual_grid_size=actual_grid_size).to(device)
    target_network.load_state_dict(q_network.state_dict()) #* copies weights from online to target

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    episode_count = 0  # Track total number of completed episodes
    
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
    
    global_update_step = 0
    update_credits = 0.0
    next_target_update = args.target_network_frequency
    for iteration in range(0, args.total_timesteps, args.num_envs):
        global_step = iteration
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
                    ep_return = info['episode']['r']
                    episode_return = ep_return.item() if hasattr(ep_return, "item") else float(ep_return[0] if isinstance(ep_return, (list, np.ndarray)) else ep_return)
                    ep_length = info['episode']['l']
                    episode_length = ep_length.item() if hasattr(ep_length, "item") else int(ep_length[0] if isinstance(ep_length, (list, np.ndarray)) else ep_length)
                    
                    recent_returns.append(episode_return)
                    
                    # Reduce logging frequency
                    if episode_count % 100 == 0:
                        print(f"global_step={global_step}, episode={episode_count}, episodic_return={episode_return:.2f}, episodic_length={episode_length}")
                        writer.add_scalar("charts/episodic_return", episode_return, global_step)
                        writer.add_scalar("charts/episodic_length", episode_length, global_step)
                        writer.add_scalar("charts/moving_avg_return", np.mean(recent_returns), global_step)
                        writer.add_scalar("charts/episodic_return_by_episode", episode_return, episode_count)

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
        if global_step > args.learning_starts: #* don't train until we have a certain number of transitions into the buffer
            # Accumulate fractional updates: UTD controls gradient steps per env step
            # e.g., utd=1.0 → 1 update/step, utd=0.1 → 1 update/10 steps, utd=2.0 → 2 updates/step
            update_credits += args.num_envs * args.utd_ratio
            num_updates = int(update_credits)
            update_credits -= num_updates
            for _ in range(num_updates):
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
                    
                    #* 6. Apply reward rotation: e^{i * w * R} in the frequency domain corresponds to shifting the distribution by R in the spatial domain.
                    reward_rotation = torch.exp(1j * omega_grid.view(1, -1) * data.rewards)
                    
                    #* 7. Bellman target, then project onto valid distributions via IFFT cleaning
                    td_target_complex_scalar = reward_rotation * interp_V
                    td_target_complex_scalar = get_cleaned_target_cf(omega_grid, td_target_complex_scalar, q_min=args.q_min, q_max=args.q_max)

                current_Q_complex_all = q_network(data.observations)
                current_V_complex_scalar = current_Q_complex_all[batch_idx, data.actions.flatten()]
                
                #*Weighted MSE Loss in Frequency Domain with Gaussian Weights 
                #! the weights and parameters were left to be tuned (kept the one that were given initially)
                sigma = 0.3
                weights = torch.exp(-(omega_grid ** 2) / (2 * sigma ** 2))
                weights = weights / weights.sum() #* normalize weights to keep loss scale consistent regardless of K or W
                unweighted_mse = torch.abs(current_V_complex_scalar - td_target_complex_scalar) ** 2
                
                weighted_mse = torch.sum(weights.view(1, -1) * unweighted_mse, dim=1) #TODO: check weights.view(1, -1)
                loss = torch.mean(weighted_mse)

                # Ensure logging frequency is tied to the UTD scaling and environment steps
                log_freq_scaled = max(1, int(10000 * args.utd_ratio))
                if global_update_step % log_freq_scaled == 0: #! used to be 100
                    writer.add_scalar("losses/loss", loss.item(), global_step)
                    
                    current_Q_all = ifft_collapse_q_values(omega_grid, current_Q_complex_all, q_min=args.q_min, q_max=args.q_max)
                    current_Q_taken = current_Q_all[batch_idx, data.actions.flatten()]
                    
                    writer.add_scalar("losses/q_values", current_Q_taken.mean().item(), global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    
                    with torch.no_grad():
                        target_V_diag = target_network(data.observations)
                        target_Q_diag = ifft_collapse_q_values(omega_grid, target_V_diag, q_min=args.q_min, q_max=args.q_max)
                        target_Q_taken_diag = target_Q_diag[batch_idx, data.actions.flatten()]
                        writer.add_scalar("diagnostics/target_q_values", target_Q_taken_diag.mean().item(), global_step)
                        
                        online_target_diff = (current_Q_all - target_Q_diag).abs().mean()
                        writer.add_scalar("diagnostics/q_online_target_diff", online_target_diff.item(), global_step)
                        
                        q_sorted = current_Q_all.sort(dim=1, descending=True).values
                        action_gap = (q_sorted[:, 0] - q_sorted[:, 1]).mean()
                        writer.add_scalar("diagnostics/action_gap", action_gap.item(), global_step)
                        
                        max_q_est = current_Q_all.abs().max().item()
                        writer.add_scalar("diagnostics/max_q_magnitude", max_q_est, global_step)
                    
                    #! Current diagnostics (temporary)

                    # === SUSPECT 1: V(0) normalization health ===
                    # If min |V(0)| drops near 0, the division blows up the CF
                    writer.add_scalar("diagnostics/v_at_zero_mag_min", q_network._v_at_zero_mag.min().item(), global_step)
                    writer.add_scalar("diagnostics/v_at_zero_mag_mean", q_network._v_at_zero_mag.mean().item(), global_step)
                    # How much magnitude clipping was needed (>1 means network wants invalid CFs)
                    writer.add_scalar("diagnostics/pre_clamp_max_magnitude", q_network._pre_clamp_max_mag.item(), global_step)
                    
                    # === SUSPECT 2: Phase interpolation health ===
                    with torch.no_grad():
                        target_phases = torch.angle(target_V_complex_all[batch_idx, next_actions])
                        phase_diffs = torch.diff(target_phases, dim=-1)
                        max_phase_jump = phase_diffs.abs().max().item()
                        mean_phase_jump = phase_diffs.abs().mean().item()
                        writer.add_scalar("diagnostics/phase_max_jump", max_phase_jump, global_step)
                        writer.add_scalar("diagnostics/phase_mean_jump", mean_phase_jump, global_step)
                    
                    # === SUSPECT 3: Spatial mask clipping ===
                    with torch.no_grad():
                        # Compare masked vs unmasked Q-values to see if the mask is clipping real mass
                        q_masked, q_unmasked, pdf_unmasked = ifft_collapse_q_values(
                            omega_grid, current_Q_complex_all, q_min=args.q_min, q_max=args.q_max, return_diagnostics=True
                        )
                        mask_bias = (q_unmasked - q_masked).abs().mean().item()
                        writer.add_scalar("diagnostics/mask_bias", mask_bias, global_step)
                        # What fraction of PDF mass falls outside [q_min, q_max]?
                        K = current_Q_complex_all.shape[-1]
                        W = torch.abs(omega_grid[0]).item()
                        dx = math.pi / W
                        x_grid = torch.linspace(-(K // 2) * dx, (K // 2 - 1) * dx, K, device=device)
                        valid_mask = (x_grid >= args.q_min) & (x_grid <= args.q_max)
                        mass_outside = (pdf_unmasked * (~valid_mask).float()).sum(dim=-1).mean().item()
                        writer.add_scalar("diagnostics/pdf_mass_outside_support", mass_outside, global_step)
                    
                optimizer.zero_grad()
                loss.backward()
                
                # === GRADIENT HEALTH (must be after backward, before clip) ===
                if global_update_step % log_freq_scaled == 0:
                    total_grad_norm = sum(p.grad.norm().item() ** 2 for p in q_network.parameters() if p.grad is not None) ** 0.5
                    writer.add_scalar("diagnostics/grad_norm_before_clip", total_grad_norm, global_step)
                
                nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
                optimizer.step()
                global_update_step += 1

            #* Target network update — both modes run every env step, matching original behavior
            if args.target_network_frequency > 0 and global_step >= next_target_update:
                #* Hard update: copy online -> target every target_network_frequency env steps
                target_network.load_state_dict(q_network.state_dict())
                next_target_update += args.target_network_frequency
            elif args.target_network_frequency == 0:
                #* Soft Polyak update every env step.
                #* Scale tau by num_envs: with N parallel envs, this update fires once
                #* per N transitions, so we need N× the per-step tau to match the
                #* effective tracking rate of the single-env case.
                effective_tau = min(args.tau * args.num_envs, 1.0)
                for target_param, param in zip(target_network.parameters(), q_network.parameters()):
                    target_param.data.copy_(effective_tau * param.data + (1.0 - effective_tau) * target_param.data)

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
    if args.profile:
        prof.stop()
