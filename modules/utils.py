import torch
import matplotlib.pyplot as plt
import numpy as np

class PreAllocatedReplayBuffer:
    def __init__(self, capacity, num_agents, obs_dim, act_dim, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate massive continuous blocks of VRAM
        self.obs = torch.zeros((capacity, num_agents, obs_dim), device=device)
        self.actions = torch.zeros((capacity, num_agents, act_dim), device=device)
        self.rewards = torch.zeros((capacity, num_agents), device=device)
        self.next_obs = torch.zeros((capacity, num_agents, obs_dim), device=device)
        self.dones = torch.zeros((capacity, 1), device=device)

    def push(self, obs, actions, rewards, next_obs, done):
        # Insert directly into the allocated VRAM blocks (lightning fast)
        self.obs[self.ptr] = obs.detach()
        self.actions[self.ptr] = actions.detach()
        self.rewards[self.ptr] = rewards.detach()
        self.next_obs[self.ptr] = next_obs.detach()
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        # Generate random indices directly on the GPU
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        # Slice the continuous VRAM blocks instantly
        return (
            self.obs[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_obs[idxs],
            self.dones[idxs]
        )

    def __len__(self):
        return self.size

def detect_device():
    '''
    Detect Device (CUDA for NVIDIA, MPS for Apple Silicon, CPU as fallback)
    '''
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def plot_learning_curve(config, reward_history, noise_history, save_path='learning_curve.png'):
    MOVING_AVG_WINDOW = config["training"]["moving_avg_window"]

    fig_curve, ax1 = plt.subplots(figsize=(7, 5))

    # --- Primary Y-Axis (Left) for Rewards ---
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.plot(reward_history, color='blue', alpha=0.3, label='Raw Reward')

    if len(reward_history) > MOVING_AVG_WINDOW:
        avg = np.convolve(reward_history, np.ones(MOVING_AVG_WINDOW)/MOVING_AVG_WINDOW, mode='valid')
        ax1.plot(range(MOVING_AVG_WINDOW - 1, len(reward_history)), avg, color='red', linewidth=2, label='Reward Trend')

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total Reward", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # --- Secondary Y-Axis (Right) for Noise ---
    ax2 = ax1.twinx()
    ax2.plot(noise_history, color='green', linestyle=':', linewidth=2, label='Exploration Noise')
    ax2.set_ylabel("Noise Std", color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # --- Combine Legends ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

    plt.title("Learning Curve & Exploration Noise Decay")
    plt.tight_layout()
    plt.savefig(save_path)