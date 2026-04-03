import unittest
from unittest.mock import patch
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# Import the environment
from uav_environment import MultiUAVEnv
from modules.fomation_type import FormationType

class TestFormationGeometries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up testing constants and create a directory for the plots."""
        cls.device = torch.device("cpu")
        cls.output_dir = "results/formation_test_plots"
        os.makedirs(cls.output_dir, exist_ok=True)
        print(f"\n--- Saving formation visualization plots to: {os.path.abspath(cls.output_dir)} ---\n")

    def create_mock_env(self, formation_type, amount, heading_deg, leader_pos):
        """
        Creates a MultiUAVEnv instance with a mocked configuration and injects
        the necessary state to test formation geometries mathematically.
        """
        config = {
            "simulation": {"dt": 1, "grid_size": 100, "num_steps": 200, "ref_scale": 100.0},
            "uav": {
                "amount": amount,
                "drag_coeff": 0.8,
                "max_velocity": 3.0,
                "max_accel": 1.0,
                "max_omega": 0.75,
                "logics": {
                    "formation_keeping": {
                        "type": formation_type,
                        "distance": 10.0,
                        "reward": 10.0,
                        "tolerance": 1.0,
                        "heading": "mocked_heading" # Bypasses 'Direction' dependency
                    }
                }
            }
        }
        
        env = MultiUAVEnv(config, self.device)
        
        # Manually inject state to bypass the simulation reset/physics sequence
        env.leader_idx = 0
        env.leader_pos = torch.tensor(leader_pos, dtype=torch.float32, device=self.device)
        env.agents_pos = torch.zeros((amount, 2), device=self.device)
        env.agents_pos[0] = env.leader_pos
        
        # Convert heading degrees to a target directional vector
        target_rad = np.deg2rad(heading_deg)
        env.target_vec = torch.tensor([np.cos(target_rad), np.sin(target_rad)], dtype=torch.float32, device=self.device)
        
        return env

    def extract_slots(self, env):
        """
        Intercepts the 'valid_slots' parameter from _assign_and_score_slots before
        the environment tries to map drones to them.
        """
        with patch.object(MultiUAVEnv, '_assign_and_score_slots') as mock_assign:
            # Dummy return to prevent downstream crashing
            mock_assign.return_value = torch.zeros(env.num_agents, device=env.device)
            
            rel_pos_tip = env.agents_pos - env.leader_pos.unsqueeze(0)
            
            # Trigger the respective formation calculation
            if env.formation_keeping_type == FormationType.LINE:
                env._calc_line_formation_keeping_rewards(rel_pos_tip)
            elif env.formation_keeping_type == FormationType.COLUMN:
                env._calc_column_formation_keeping_rewards(rel_pos_tip)
            elif env.formation_keeping_type == FormationType.V_SHAPE:
                env._calc_v_shape_formation_keeping_rewards(rel_pos_tip)
            elif env.formation_keeping_type == FormationType.CIRCLE:
                env._calc_circle_formation_keeping_rewards(rel_pos_tip)
            
            # Extract valid_slots from what the environment passed to _assign_and_score_slots
            valid_slots = mock_assign.call_args[0][0]
            
        return valid_slots

    def plot_formation_scenarios(self, formation_type, formation_name):
        """
        Generates a 3-panel plot showcasing the formation geometry for different 
        leader coordinates and directional headings.
        """
        # Define 3 different scenarios to test boundary translation and rotation
        scenarios = [
            {"leader_pos": [50.0, 50.0], "heading": 90.0,  "title": "Center (Heading: North)"},
            {"leader_pos": [20.0, 20.0], "heading": 45.0,  "title": "Bottom-Left (Heading: NE)"},
            {"leader_pos": [80.0, 70.0], "heading": -45.0, "title": "Top-Right (Heading: SE)"},
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Formation Geometry: {formation_name}", fontsize=16, fontweight='bold')
        
        for ax, scenario in zip(axes, scenarios):
            # 1 Leader + 4 Followers
            env = self.create_mock_env(
                formation_type=formation_type, 
                amount=5, 
                heading_deg=scenario["heading"], 
                leader_pos=scenario["leader_pos"]
            )
            
            valid_slots = self.extract_slots(env)
            
            # Convert tensors to numpy for plotting
            leader_np = env.leader_pos.numpy()
            target_np = env.target_vec.numpy()
            
            # valid_slots are generated relative to the leader, so we add the leader's position
            slots_np = valid_slots.numpy() + leader_np 
            
            # Plot Leader
            ax.scatter(leader_np[0], leader_np[1], color='gold', s=150, edgecolors='black', label='Leader', zorder=4)
            
            # Plot Heading Vector (Direction of Flight)
            ax.quiver(leader_np[0], leader_np[1], target_np[0], target_np[1], 
                      color='red', scale=8, width=0.015, label='Heading', zorder=3)
            
            # Plot Follower Target Slots
            ax.scatter(slots_np[:, 0], slots_np[:, 1], color='dodgerblue', s=100, edgecolors='black', label='Follower Slots', zorder=4)
            
            # Draw structural lines connecting leader to slots
            for slot in slots_np:
                ax.plot([leader_np[0], slot[0]], [leader_np[1], slot[1]], color='gray', linestyle='--', alpha=0.7, zorder=2)
            
            # Plot formatting
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_title(scenario["title"], fontsize=12)
            ax.set_aspect('equal')
            ax.grid(True, linestyle=':', alpha=0.6)
            if ax == axes[0]:
                ax.legend(loc='upper left')
                
        # plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{formation_name}_test.png")
        plt.savefig(save_path, dpi=150)
        plt.close()

    # --- INDIVIDUAL TESTS ---
    def test_01_line_formation(self):
        self.plot_formation_scenarios(FormationType.LINE, "LINE")

    def test_02_column_formation(self):
        self.plot_formation_scenarios(FormationType.COLUMN, "COLUMN")

    def test_03_v_shape_formation(self):
        self.plot_formation_scenarios(FormationType.V_SHAPE, "V_SHAPE")

    def test_04_circle_formation(self):
        self.plot_formation_scenarios(FormationType.CIRCLE, "CIRCLE")

if __name__ == '__main__':
    unittest.main(verbosity=2)