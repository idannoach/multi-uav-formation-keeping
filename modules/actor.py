import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, max_accel, max_omega):
        super(Actor, self).__init__()
        
        # Dynamically scale hidden nodes: roughly 4x to 8x the input size, minimum 256
        hidden_dim = max(256, obs_dim * 8)
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim) 
        
        self.action_out = nn.Linear(hidden_dim, action_dim)
        self.max_accel = max_accel
        self.max_omega = max_omega

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        raw_actions = torch.tanh(self.action_out(x))
        
        accel = (raw_actions[:, 0].unsqueeze(1) + 1.0) / 2.0 * self.max_accel
        omega = raw_actions[:, 1].unsqueeze(1) * self.max_omega
        
        return torch.cat([accel, omega], dim=1)