import torch
import torch.nn.functional as F
import torch.optim as optim
import json
import os
import logging
from tqdm import tqdm
import subprocess
import datetime

from modules.actor import Actor
from modules.critic import Critic
from modules.utils import PreAllocatedReplayBuffer, detect_device, plot_learning_curve
from uav_environment import MultiUAVEnv
from modules.fomation_type import FormationType

def init_logging(formation_type):
    os.makedirs(f"results/{formation_type}", exist_ok=True)

    # --- 1. Clear existing log handlers ---
    # This forces Python to let go of the previous log file in the loop
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # --- 2. Generate a unique timestamp ---
    timestamp = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    log_filename = f'results/{formation_type}/training_log_{timestamp}.txt'

    # --- 3. Initialize the new unique logger ---
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s | %(message)s', 
        filename=log_filename, 
        filemode='w'
    )

def train_maddpg(env, config, device):
    '''
    Core MADDPG Training Loop with Decoupled Actor Updates and Pre-Allocated Replay Buffer
    '''
    # ==========================================
    # CONFIGURTAION AND HYPERPARAMETERS
    # ==========================================
    num_epochs = config["training"]["num_epochs"]
    num_steps = config["simulation"]["num_steps"]
    gamma = config["training"]["gamma"]
    lr = config["training"]["lr"]
    num_agents = config["uav"]["amount"]

    action_dim = config["training"]["action_dim"]
    max_accel = config["uav"]["max_accel"]
    max_omega = config["uav"]["max_omega"]

    batch_size = config["training"]["batch_size"]
    updates_per_epoch = config["training"]["updates_per_epoch"]
    tau = config["training"]["tau"]
    
    noise_std = config["training"]["noise_std"]
    noise_decay = config["training"]["noise_decay"]
    min_noise = config["training"]["min_noise"]

    formation_type = FormationType(config["uav"]["logics"]["formation_keeping"]["type"]).name

    # Run a dummy reset to get the observation tensor
    dummy_obs = env.reset()
    # Read the size of the second dimension (obs_dim)
    obs_dim = dummy_obs.shape[1]

    buffer = PreAllocatedReplayBuffer(capacity=100000, num_agents=num_agents, obs_dim=obs_dim, act_dim=action_dim, device=device)

    # ==========================================
    # NETWORK INITIALIZATION
    # ==========================================

    # Initialize ONE Leader Actor and ONE Shared Follower Actor
    leader_actor = Actor(obs_dim, action_dim, max_accel, max_omega).to(device)
    target_leader_actor = Actor(obs_dim, action_dim, max_accel, max_omega).to(device)
    target_leader_actor.load_state_dict(leader_actor.state_dict())

    follower_actor = Actor(obs_dim, action_dim, max_accel, max_omega).to(device)
    target_follower_actor = Actor(obs_dim, action_dim, max_accel, max_omega).to(device)
    target_follower_actor.load_state_dict(follower_actor.state_dict())

    # Keep 5 separate Critics to evaluate the 5 individual slots
    critics = [Critic(num_agents, obs_dim, action_dim).to(device) for _ in range(num_agents)]
    target_critics = [Critic(num_agents, obs_dim, action_dim).to(device) for _ in range(num_agents)]
    
    for i in range(num_agents):
        target_critics[i].load_state_dict(critics[i].state_dict())

    # ==========================================
    # TRAINING LOOP
    # ==========================================
    reward_history = []
    noise_history = []

    logging.info(f"Starting Training for Formation: {formation_type} on {device}")
    pbar = tqdm(range(num_epochs), desc=f"Training {formation_type}")

    leader_optimizer = optim.Adam(leader_actor.parameters(), lr=lr)
    follower_optimizer = optim.Adam(follower_actor.parameters(), lr=lr)
    critic_optimizers = [optim.Adam(c.parameters(), lr=lr) for c in critics]

    for epoch in pbar:
        obs = env.reset()
        epoch_reward = 0.0
        
        # ==========================================
        # PHASE 1: ROUTED DATA COLLECTION
        # ==========================================
        # Define the physical limits so exploration noise is proportional
        action_scale = torch.tensor([env.max_accel, env.max_omega], device=device)
        
        for step in range(num_steps):
            actions_list = []
            
            with torch.no_grad():
                for i in range(num_agents):
                    obs_tensor = obs[i].unsqueeze(0) 
                    
                    if i == env.leader_idx:
                        action = leader_actor(obs_tensor).squeeze(0)
                    else:
                        action = follower_actor(obs_tensor).squeeze(0)
                        
                    # Normalized Action Noise
                    # Generates a standard normal curve, scales it by the decay factor, 
                    # and then stretches it to fit the exact physical limits of the drone.
                    noise = torch.randn_like(action) * noise_std * action_scale
                    action = action + noise
                    
                    action[0] = torch.clamp(action[0], 0, env.max_accel)
                    action[1] = torch.clamp(action[1], -env.max_omega, env.max_omega)
                    actions_list.append(action)
            
            actions = torch.stack(actions_list)
            next_obs, rewards = env.step(actions.detach())
            done = step == num_steps - 1
            
            buffer.push(obs, actions, rewards, next_obs, done)
            obs = next_obs
            epoch_reward += rewards.sum().item()

        # ==========================================
        # PHASE 2: DECOUPLED NETWORK UPDATES
        # ==========================================
        if len(buffer) > batch_size:
            for _ in range(updates_per_epoch):
                b_obs, b_actions, b_rewards, b_next_obs, b_dones = buffer.sample(batch_size)
                b_rewards = b_rewards * 0.1 
                
                # --- 1. UPDATE CRITICS ---
                for i in range(num_agents):
                    with torch.no_grad():
                        next_actions = []
                        for j in range(num_agents):
                            if j == env.leader_idx:
                                next_actions.append(target_leader_actor(b_next_obs[:, j, :]))
                            else:
                                next_actions.append(target_follower_actor(b_next_obs[:, j, :]))
                                
                        target_q = target_critics[i]([b_next_obs[:, j, :] for j in range(num_agents)], next_actions)
                        y = b_rewards[:, i].unsqueeze(1) + gamma * target_q 
                    
                    current_q = critics[i]([b_obs[:, j, :] for j in range(num_agents)], [b_actions[:, j, :] for j in range(num_agents)])
                    critic_loss = F.mse_loss(current_q, y)
                    
                    critic_optimizers[i].zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critics[i].parameters(), 0.5)
                    critic_optimizers[i].step()
                    
                # --- 2. PRE-CALCULATE ACTOR OUTPUTS ---
                leader_optimizer.zero_grad()
                follower_optimizer.zero_grad()
                
                all_actions = []
                for j in range(num_agents):
                    if j == env.leader_idx:
                        all_actions.append(leader_actor(b_obs[:, j, :]))
                    else:
                        all_actions.append(follower_actor(b_obs[:, j, :]))
                
                # --- 3. UPDATE LEADER ACTOR (Cooperative: All Critics) ---
                leader_loss_total = 0
                
                # We only need to detach the followers' actions once
                curr_actions_leader = [all_actions[0]] + [a.detach() for a in all_actions[1:]]
                
                # Sum the loss from ALL 5 critics to force cooperative behavior
                for i in range(num_agents):
                    leader_loss_total += -critics[i]([b_obs[:, j, :] for j in range(num_agents)], curr_actions_leader).mean()
                
                # Average the loss
                leader_loss = leader_loss_total / num_agents
                
                leader_loss.backward()
                torch.nn.utils.clip_grad_norm_(leader_actor.parameters(), 0.5)
                leader_optimizer.step()
                
                # --- 4. UPDATE FOLLOWER ACTOR (Average of Critics 1-4) ---
                follower_loss_total = 0
                for i in range(1, num_agents):
                    curr_actions_follower = []
                    for j in range(num_agents):
                        if j == i:
                            curr_actions_follower.append(all_actions[j])
                        else:
                            curr_actions_follower.append(all_actions[j].detach())
                    follower_loss_total += -critics[i]([b_obs[:, j, :] for j in range(num_agents)], curr_actions_follower).mean()
                
                follower_loss = follower_loss_total / (num_agents - 1)
                follower_loss.backward()
                torch.nn.utils.clip_grad_norm_(follower_actor.parameters(), 0.5)
                follower_optimizer.step()
                
                # --- 5. SOFT UPDATE TARGET NETWORKS ---
                for param, target_param in zip(leader_actor.parameters(), target_leader_actor.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(follower_actor.parameters(), target_follower_actor.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for i in range(num_agents):
                    for param, target_param in zip(critics[i].parameters(), target_critics[i].parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # ==========================================
        # PHASE 3: LOGGING AND NOISE DECAY
        # ==========================================
        noise_std = max(min_noise, noise_std * noise_decay)
        reward_history.append(epoch_reward)
        noise_history.append(noise_std)

        pbar.set_postfix({"Reward": f"{epoch_reward:.2f}", "Noise": f"{noise_std:.3f}"})
        logging.info(f"Epoch {epoch} | Total Reward: {epoch_reward:.2f} | Noise Std: {noise_std:.3f}")

    return leader_actor, follower_actor, reward_history, noise_history

if __name__ == "__main__":
    subprocess.run("cls", shell=True, check=True)
    
    for config_file in os.listdir('configs'):
        if config_file.endswith('.json'):
            config_path = os.path.join('configs', config_file)
            with open(config_path, 'r') as f:
                config = json.load(f)

                device = detect_device()

                env = MultiUAVEnv(config, device=device)

                formation_type = FormationType(config["uav"]["logics"]["formation_keeping"]["type"]).name

                os.makedirs(f"results/{formation_type}", exist_ok=True)

                init_logging(formation_type)

                # Pass the decoupled actors to the training function
                trained_leader, trained_follower, reward_history, noise_history = train_maddpg(env, config, device)

                # Save the brains
                torch.save(trained_leader.state_dict(), f"results/{formation_type}/trained_leader.pth")
                torch.save(trained_follower.state_dict(), f"results/{formation_type}/trained_follower.pth")

                # Plot the learning curve
                plot_learning_curve(config, reward_history, noise_history, save_path=f"results/{formation_type}/learning_curve.png")
