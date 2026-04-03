import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, num_agents, obs_dim, action_dim):
        super(Critic, self).__init__()
        
        total_obs_dim = obs_dim * num_agents
        total_action_dim = action_dim * num_agents
        input_dim = total_obs_dim + total_action_dim
        
        # The Critic needs to be massive for large swarms. 
        # Make the hidden layer at least 1.5x to 2x the input dimension.
        hidden_dim = max(256, int(input_dim * 1.5))
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        self.q_out = nn.Linear(hidden_dim, 1)

    def forward(self, state_list, action_list):
        state = torch.cat(state_list, dim=1)
        action = torch.cat(action_list, dim=1)
        x = torch.cat([state, action], dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        return self.q_out(x)