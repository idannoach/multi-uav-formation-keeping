import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

from modules.fomation_type import FormationType
from modules.direction import Direction

class MultiUAVEnv:
    '''
    Multi-UAV Formation Keeping Environment
    '''
    def __init__(self, config, device):
        '''
        Multi-UAV Formation Keeping Environment
        '''
        # Store device
        self.device = device

        # Simulation Parameters
        self.dt = config["simulation"]["dt"]
        self.grid_size = config["simulation"]["grid_size"]
        self.half_grid = self.grid_size / 2.0
        self.num_steps = config["simulation"]["num_steps"]
        self.ref_scale = config["simulation"]["ref_scale"]

        # UAV Parameters
        self.num_agents = config["uav"]["amount"]
        self.drag_coeff = config["uav"]["drag_coeff"]
        self.max_velocity = config["uav"]["max_velocity"]
        self.max_accel = config["uav"]["max_accel"]
        self.max_omega = config["uav"]["max_omega"]
        
        # Formation Keeping Parameters
        self.formation_keeping_type = config["uav"]["logics"]["formation_keeping"]["type"]
        self.formation_keeping_distance = config["uav"]["logics"]["formation_keeping"]["distance"]
        self.formation_keeping_reward = config["uav"]["logics"]["formation_keeping"]["reward"]
        self.formation_keeping_tolerance = config["uav"]["logics"]["formation_keeping"]["tolerance"]
        self.formation_keeping_heading = config["uav"]["logics"]["formation_keeping"]["heading"]

    def reset(self):
        '''
        Reset the environment to an initial state and return the initial observation
        '''
        # Initialize step counter
        self.current_step = 0

        # Initialize agents in random positions and headings, with zero velocity
        self._reset_agents()

        # Initialize leader index and position
        self._reset_leader()

        return self._get_obs()
    
    def _reset_agents(self, spawn_near_center = True):
        '''
        Initialize agents in random positions and headings, with zero velocity
        '''
        # Position Initialization: Option to spawn near center or randomly across the grid
        if spawn_near_center:
            center_spawn = torch.tensor([self.grid_size * 0.2, self.grid_size * 0.5], device=self.device)
            noise = torch.empty(self.num_agents, 2, device=self.device).uniform_(-20, 20)
            self.agents_pos = center_spawn + noise
        else:
            self.agents_pos = torch.empty(self.num_agents, 2, device=self.device).uniform_(0, self.grid_size)

        # Velocity Initialization: Start at rest
        self.agents_vel = torch.zeros((self.num_agents, 2), device=self.device)

        # Heading Initialization: Random initial headings
        self.agents_theta = torch.empty(self.num_agents, device=self.device).uniform_(0, 2 * np.pi)

    def _reset_leader(self):
        '''
        STATIC LEADER: Locks the leader to Agent 0. 
        This is strictly required for MADDPG Critics to successfully map the concatenated state array without permutation confusion.
        '''
        # Hardcode Agent 0 as the permanent anchor
        self.leader_idx = 0
        
        # Pre-compute Target Vectors on device
        target_rad = np.deg2rad(Direction(self.formation_keeping_heading).heading_degrees())
        self.target_vec = torch.tensor([np.cos(target_rad), np.sin(target_rad)], device=self.device, dtype=torch.float32)
        
        self.leader_pos = self.agents_pos[self.leader_idx]

    def _get_obs(self):
        '''
        Observation Structure (Per Agent):
        [Normalized Velocity (2), Heading Vector (2), Target Vector (2),
          Relative Position to Leader (2), Relative Positions to Other Agents (2*(N-1)),
            Relative Velocities to Other Agents (2*(N-1))]
        '''
        # Self-awareness: Velocity, Heading, Target Vector
        norm_vel, heading_vec, target_obs = self._get_self_awareness_obs()
        
        # Leader awareness: Relative Position to Leader
        norm_rel_pos_leader = self._get_leader_awareness_obs()

        # Neighbor awareness: Relative Positions and Velocities to Other Agents
        other_agents_obs = self._get_other_agents_obs()
        
        # Concatenate all observations into a single tensor of shape (num_agents, obs_dim)
        obs = torch.cat([norm_vel, heading_vec, target_obs, norm_rel_pos_leader, other_agents_obs], dim=1)
        return obs
    
    def _get_self_awareness_obs(self):
        '''
        Agents self-awareness (Vel, Heading) and Target Vector (Global Reference)
         - Velocity is normalized by max_velocity to keep it in a consistent range for the Actor network
        '''
        # Velocity and Heading
        norm_vel = self.agents_vel / self.max_velocity
        heading_vec = torch.stack([torch.cos(self.agents_theta), torch.sin(self.agents_theta)], dim=1)
        
        # Target Vector (Same for all agents, repeated to match batch size)
        target_obs = self.target_vec.repeat(self.num_agents, 1)

        return norm_vel, heading_vec, target_obs
    
    def _get_leader_awareness_obs(self):
        '''
        Leader awareness 
        '''
        # Relative Position to Leader
        rel_pos_leader = self.leader_pos.unsqueeze(0) - self.agents_pos
        rel_pos_leader = (rel_pos_leader + self.half_grid) % self.grid_size - self.half_grid
        norm_rel_pos_leader = rel_pos_leader / self.ref_scale
        return norm_rel_pos_leader
    
    def _get_other_agents_obs(self):
        '''
        Relative states of OTHER agents (Positions and Velocities)
         - Relative positions are normalized by ref_scale to keep them in a consistent range for the Actor network
        '''
        # Relative Positions and Velocities to Other Agents
        rel_pos_matrix = self.agents_pos.unsqueeze(1) - self.agents_pos.unsqueeze(0)
        rel_pos_matrix = (rel_pos_matrix + self.half_grid) % self.grid_size - self.half_grid
        norm_rel_pos_matrix = rel_pos_matrix / self.ref_scale
        
        # Relative Velocities to Other Agents
        rel_vel_matrix = (self.agents_vel.unsqueeze(1) - self.agents_vel.unsqueeze(0)) / self.max_velocity
        
        # We only want the relative positions and velocities to OTHER agents,
        #  so we mask out the self-relations and flatten the rest into a single vector per agent.
        #  This results in (num_agents, 2*(N-1))
        other_obs_list = []
        for i in range(self.num_agents):
            mask = torch.arange(self.num_agents, device=self.device) != i
            other_pos = norm_rel_pos_matrix[i][mask].flatten()
            other_vel = rel_vel_matrix[i][mask].flatten()
            other_obs_list.append(torch.cat([other_pos, other_vel]))
        other_agents_obs = torch.stack(other_obs_list)

        return other_agents_obs

    def step(self, actions):
        '''
        Apply the given actions, update the environment state, and return the new observation and rewards.
        '''
        # Update current step counter
        self.current_step += 1

        # Update Physics
        self._update_physics(actions)

        # Update leader position
        self.leader_pos = self.agents_pos[self.leader_idx]

        return self._get_obs(), self._calc_formation_keeping_rewards()
    
    def _update_physics(self, actions):
        '''
        Update the physics of the environment based on the actions taken by the agents.
        Actions are expected to be in the range [-1, 1] and will be scaled to the appropriate accel and omega ranges.
        '''
        # Scale actions to actual accel and omega values
        accel_cmds = actions[:, 0]
        omega_cmds = actions[:, 1]
        
        # Update Theta
        self.agents_theta += omega_cmds * self.dt
        self.agents_theta = (self.agents_theta + np.pi) % (2 * np.pi) - np.pi

        # Thrust Vector
        cos_theta = torch.cos(self.agents_theta)
        sin_theta = torch.sin(self.agents_theta)
        thrust_vec = torch.stack([cos_theta, sin_theta], dim=1) * accel_cmds.unsqueeze(1)
        
        # Drag and Total Accel
        drag_vec = -self.drag_coeff * self.agents_vel
        total_accel = thrust_vec + drag_vec
        
        # Update Velocity
        self.agents_vel += total_accel * self.dt
        speeds = torch.norm(self.agents_vel, dim=1, keepdim=True)
        self.agents_vel = torch.where(
            speeds > self.max_velocity,
            (self.agents_vel / (speeds + 1e-6)) * self.max_velocity,
            self.agents_vel
        )
        
        # Update Position with Wrap Around (Torus)
        self.agents_pos += self.agents_vel * self.dt
        self.agents_pos = self.agents_pos % self.grid_size

    def _calc_formation_keeping_rewards(self):
        '''
        Formation Keeping Reward combines multiple factors to encourage agents to maintain a cohesive formation:
        1. Formation Type Keeping: Rewards agents for maintaining the correct geometric formation type (LINE, COLUMN, V_SHAPE, CIRCLE).
            This encourages agents to maintain the overall shape of the formation, rather than just spacing.    
        2. Formation Distance Keeping: Rewards agents for maintaining proper spacing from neighbors (not too close, not too far). 
            This encourages agents to maintain a cohesive formation without drifting apart or colliding.
        3. Formation Velocity Keeping: Rewards agents for maintaining a velocity that allows the formation to keep up with the leader's target speed. 
            This encourages followers to maintain a speed that allows them to keep up with the leader, rather than lagging behind.
        '''
        formation_keeping_rewards = torch.zeros(self.num_agents, device=self.device)

        formation_type_keeping_rewards = self._calc_formation_type_keeping_rewards()

        formation_distance_keeping_rewards = self._calc_formation_distance_keeping_rewards()

        formation_velocity_keeping_rewards = self._calc_formation_velocity_keeping_rewards()

        formation_keeping_rewards += formation_type_keeping_rewards + formation_distance_keeping_rewards + formation_velocity_keeping_rewards

        return formation_keeping_rewards
    
    def _calc_formation_type_keeping_rewards(self):
        '''
        Reward for maintaining the correct geometric formation type (LINE, COLUMN, V_SHAPE, CIRCLE).
        This encourages agents to maintain the overall shape of the formation, rather than just spacing.
        '''
        # Leader Relative Positions
        rel_pos_tip = self.agents_pos - self.leader_pos.unsqueeze(0)
        rel_pos_tip = (rel_pos_tip + self.half_grid) % self.grid_size - self.half_grid

        if self.formation_keeping_type == FormationType.LINE:
            return self._calc_line_formation_keeping_rewards(rel_pos_tip)
            
        elif self.formation_keeping_type == FormationType.COLUMN:
            return self._calc_column_formation_keeping_rewards(rel_pos_tip)

        elif self.formation_keeping_type == FormationType.V_SHAPE:
            return self._calc_v_shape_formation_keeping_rewards(rel_pos_tip)

        elif self.formation_keeping_type == FormationType.CIRCLE:
            return self._calc_circle_formation_keeping_rewards(rel_pos_tip)
        else:
            return torch.zeros(self.num_agents, device=self.device)
    
    def _calc_line_formation_keeping_rewards(self, rel_pos_tip):
        '''
        LINE: Generates slots alternating left and right of the leader.
        '''
        num_followers = self.num_agents - 1
        d = self.formation_keeping_distance
        
        # Lateral vector is perpendicular to target vector [-Ty, Tx]
        lat_dir = torch.tensor([-self.target_vec[1], self.target_vec[0]], device=self.device)
        
        f_indices = torch.arange(num_followers, device=self.device)
        rows = (f_indices // 2) + 1.0
        
        # Evens go right (+1), Odds go left (-1)
        signs = ((f_indices % 2 == 0).float() * 2.0) - 1.0
        
        valid_slots = (rows * d * signs).unsqueeze(1) * lat_dir
        
        return self._assign_and_score_slots(valid_slots, rel_pos_tip)

    def _calc_column_formation_keeping_rewards(self, rel_pos_tip):
        '''
        COLUMN: Generates slots in a straight line directly behind the leader.
        '''
        num_followers = self.num_agents - 1
        d = self.formation_keeping_distance
        
        rows = torch.arange(1, num_followers + 1, device=self.device).float()
        
        # Slots are placed backwards (-target_vec)
        valid_slots = rows.unsqueeze(1) * d * (-self.target_vec)
        
        return self._assign_and_score_slots(valid_slots, rel_pos_tip)
    
    def _calc_v_shape_formation_keeping_rewards(self, rel_pos_tip):
        '''
        V_SHAPE: Generates slots on two swept-back arms.
        '''
        num_followers = self.num_agents - 1
        d = self.formation_keeping_distance
        
        target_rad = torch.atan2(self.target_vec[1], self.target_vec[0])
        sweep_angle = 3.0 * np.pi / 4.0 
        
        arm1_dir = torch.tensor([torch.cos(target_rad + sweep_angle), torch.sin(target_rad + sweep_angle)], device=self.device)
        arm2_dir = torch.tensor([torch.cos(target_rad - sweep_angle), torch.sin(target_rad - sweep_angle)], device=self.device)
        
        f_indices = torch.arange(num_followers, device=self.device)
        rows = (f_indices // 2) + 1.0
        
        even_mask = (f_indices % 2 == 0).unsqueeze(1).float()
        odd_mask = 1.0 - even_mask
        arms_matrix = even_mask * arm1_dir + odd_mask * arm2_dir
        
        valid_slots = (rows * d).unsqueeze(1) * arms_matrix
        
        return self._assign_and_score_slots(valid_slots, rel_pos_tip)
    
    def _calc_circle_formation_keeping_rewards(self, rel_pos_tip):
        '''
        CIRCLE: Generates slots evenly distributed on a ring around the leader.
        Note: The leader is the center of the circle to maintain anchor stability.
        '''
        num_followers = self.num_agents - 1
        d = self.formation_keeping_distance
        
        # Divide the circle into equal angular slices
        angles = torch.arange(num_followers, device=self.device).float() * (2 * np.pi / num_followers)
        
        # Offset angles by the target heading so the circle rotates smoothly with the flight path
        target_rad = torch.atan2(self.target_vec[1], self.target_vec[0])
        angles += target_rad
        
        valid_slots = d * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        
        return self._assign_and_score_slots(valid_slots, rel_pos_tip)

    def _assign_and_score_slots(self, valid_slots, rel_pos_tip):
        '''
        Universal Vectorized Bipartite Matching & Scoring.
        Takes geometric slots, assigns followers 1-to-1 to prevent collisions,
        and calculates a continuous global gradient score (Torus-safe).
        '''
        num_followers = self.num_agents - 1
        formation_type_keeping_rewards = torch.zeros(self.num_agents, device=self.device)
        formation_type_keeping_rewards[self.leader_idx] = 0.0 # Leader gets no reward for slot-keeping, only velocity-keeping

        if num_followers == 0:
            return formation_type_keeping_rewards

        follower_mask = torch.arange(self.num_agents, device=self.device) != self.leader_idx
        follower_indices = torch.where(follower_mask)[0]
        rel_pos_followers = rel_pos_tip[follower_indices]

        # --- BUG FIX: Torus-Safe Distance Matrix ---
        # Calculate raw differences, wrap them around the Torus, then take the norm
        diff_matrix = rel_pos_followers.unsqueeze(1) - valid_slots.unsqueeze(0)
        diff_matrix = (diff_matrix + self.half_grid) % self.grid_size - self.half_grid
        dist_matrix = torch.norm(diff_matrix, dim=2)

        assigned_slots = torch.zeros(num_followers, dtype=torch.long, device=self.device)
        dist_matrix_clone = dist_matrix.clone()

        for _ in range(num_followers):
            min_idx = torch.argmin(dist_matrix_clone)
            f_idx = min_idx // num_followers
            s_idx = min_idx % num_followers
            
            assigned_slots[f_idx] = s_idx
            
            dist_matrix_clone[f_idx, :] = float('inf')
            dist_matrix_clone[:, s_idx] = float('inf')

        assigned_slot_pos = valid_slots[assigned_slots]
        
        # --- BUG FIX: Torus-Safe Score Calculation ---
        # Wrap the difference between the follower and its assigned slot
        diff = rel_pos_followers - assigned_slot_pos
        diff = (diff + self.half_grid) % self.grid_size - self.half_grid
        dist_to_slot = torch.norm(diff, dim=1)
        
        # --- THE FIX: The "Gravity Well" Reward ---
        # 1. Global Pull (20% of points): A weak, linear gradient so they never get lost across the map
        global_pull = torch.clamp(1.0 - (dist_to_slot / self.half_grid), min=0.0)
        
        # 2. Local Docking Pull (80% of points): An exponential curve using tolerance.
        # If they are 0m away, they get 100%. If they are just 2m away, this value collapses to ~13%.
        local_pull = torch.exp(-dist_to_slot / self.formation_keeping_tolerance)
        
        scores = (global_pull * 0.2) + (local_pull * 0.8)
        
        formation_type_keeping_rewards[follower_indices] = self.formation_keeping_reward * scores
            
        return formation_type_keeping_rewards
    
    def _calc_formation_distance_keeping_rewards(self):
        '''
        Reward for maintaining proper spacing from neighbors (not too close, not too far).
        This encourages agents to maintain a cohesive formation without drifting apart or colliding.
        '''
        # Torus-Safe Distance Matrix for Collision Detection
        # We need this so drones "feel" each other across the map boundaries
        with torch.no_grad():
            diff = self.agents_pos.unsqueeze(1) - self.agents_pos.unsqueeze(0)
            diff = (diff + self.half_grid) % self.grid_size - self.half_grid
            dist_matrix = torch.norm(diff, dim=2)
            # Ignore self-distance
            dist_matrix += torch.eye(self.num_agents, device=self.device) * 100.0
            dist_nearest, _ = torch.min(dist_matrix, dim=1)

        # --- THE FIX: Safe-Bubble Repulsion Gradient ---
        # The penalty triggers ONLY if drones breach 75% of the intended formation distance.
        # This prevents the Repulsion field from pushing them out of their perfect geometric slots.
        safe_dist = self.formation_keeping_distance * 0.75
        
        repulsion_breach = torch.clamp(safe_dist - dist_nearest, min=0)
        
        # Creates a linear slope: 0 penalty at safe_dist, climbing to -10.0 penalty if they collide.
        repulsion_penalty = (repulsion_breach / safe_dist) * 10.0
        
        # Follower-only penalty (Leader stays steady as the anchor)
        follower_mask = torch.ones(self.num_agents, device=self.device)
        follower_mask[self.leader_idx] = 0.0
        
        # Return as a negative value since this is a penalty
        return - (repulsion_penalty * follower_mask)
    

    def _calc_formation_velocity_keeping_rewards(self):
        '''
        Reward for maintaining a velocity that allows the formation to keep up with the leader's target speed.
        This encourages followers to maintain a speed that allows them to keep up with the leader, rather than lagging behind.
        '''
        formation_velocity_keeping_rewards = torch.zeros(self.num_agents, device=self.device)

        terminal_velocity = self.max_accel / self.drag_coeff
        actual_max_vel = min(self.max_velocity, terminal_velocity)
        
        # Leader cruises at 75% throttle
        cruise_speed = actual_max_vel * 0.75
        target_velocity_vec = self.target_vec * cruise_speed
        
        # Calculate the error between current velocity and the perfect cruise velocity
        leader_vel_error = torch.norm(self.agents_vel[self.leader_idx] - target_velocity_vec)
        
        # --- THE FIX: Massive Positive Velocity Reward ---
        # 1.0 = Perfect Speed & Heading. 0.0 = Standing Still. Negative = Flying backward.
        speed_score = 1.0 - (leader_vel_error / cruise_speed)
        
        # Multiply by 20.0 to make moving the most valuable action in the entire simulation.
        # This heavily outweighs any temporary points lost by the followers lagging behind.
        formation_velocity_keeping_rewards[self.leader_idx] = 20.0 * speed_score
        
        # Followers receive 0 velocity reward. They only care about hitting their slots.
        return formation_velocity_keeping_rewards

    def render(self, epoch_number):
        '''
        Render the current state of the environment using Matplotlib.
         - Agents are shown as points with arrows indicating heading.
         - The leader is highlighted in gold.
         - Lines are drawn between agents to visualize the formation.
        '''
        
        if not hasattr(self, 'fig'):
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.ax.clear()
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        
        pos = self.agents_pos.cpu().detach().numpy()
        theta = self.agents_theta.cpu().detach().numpy()

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                self.ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], color='cyan', alpha=0.9, linestyle='--')

        colors = ['blue'] * self.num_agents
        colors[self.leader_idx] = 'gold'

        self.ax.scatter(pos[:,0], pos[:,1], c=colors, s=50, zorder=3)
        self.ax.quiver(pos[:,0], pos[:,1], np.cos(theta), np.sin(theta), color='red', scale=20, zorder=4)
        self.ax.set_title(f"Epoch: {epoch_number} | Step: {self.current_step}")
        plt.pause(0.001)