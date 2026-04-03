import torch
import subprocess
import json
import os
import time
from PIL import Image  # <-- NEW IMPORT

from uav_environment import MultiUAVEnv
from modules.fomation_type import FormationType
from modules.utils import detect_device
from modules.actor import Actor

def evaluate(config_path):
    # 1. Load Configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    device = detect_device()

    # 2. Initialize Environment
    env = MultiUAVEnv(config, device)
    num_agents = config["uav"]["amount"]
    
    # Run a dummy reset to get the observation tensor
    dummy_obs = env.reset()
    obs_dim = dummy_obs.shape[1]
    action_dim = 2
    
    max_accel = config["uav"]["max_accel"]
    max_omega = config["uav"]["max_omega"]
    formation_type = FormationType(config["uav"]["logics"]["formation_keeping"]["type"]).name

    # 3. Initialize Decoupled Actors
    leader_actor = Actor(obs_dim, action_dim, max_accel, max_omega).to(device)
    follower_actor = Actor(obs_dim, action_dim, max_accel, max_omega).to(device)

    # 4. Load Trained Weights
    leader_path = f"results/{formation_type}/trained_leader.pth"
    follower_path = f"results/{formation_type}/trained_follower.pth"

    if os.path.exists(leader_path) and os.path.exists(follower_path):
        leader_actor.load_state_dict(torch.load(leader_path, map_location=device))
        follower_actor.load_state_dict(torch.load(follower_path, map_location=device))
        print(f"Successfully loaded Leader and Follower models for {formation_type}.")
    else:
        print("Error: Trained model files not found. Check your file paths.")
        return

    leader_actor.eval()
    follower_actor.eval()

    # 5. Run Evaluation Simulation
    obs = env.reset()
    total_reward = 0
    num_steps = config["simulation"]["num_steps"]

    print(f"Starting Evaluation for {num_steps} steps...")
    
    # --- NEW: List to track our 3 screenshots ---
    saved_screenshots = []
    target_steps = [0, num_steps // 2, num_steps - 1]  # Start, Middle, End
    
    for step in range(num_steps):
        actions_list = []
        
        with torch.no_grad():
            for i in range(num_agents):
                obs_tensor = obs[i].unsqueeze(0) 
                
                # Route Agent 0 to the Leader brain, all others to the Follower brain
                if i == env.leader_idx:
                    action = leader_actor(obs_tensor).squeeze(0)
                else:
                    action = follower_actor(obs_tensor).squeeze(0)
                
                # Clip actions using PyTorch functions
                action[0] = torch.clamp(action[0], 0, env.max_accel)
                action[1] = torch.clamp(action[1], -env.max_omega, env.max_omega)
                
                actions_list.append(action)

        actions = torch.stack(actions_list)
        next_obs, rewards = env.step(actions)
        
        obs = next_obs
        total_reward += rewards.sum().item()

        # Render the formation
        env.render(epoch_number="EVAL")
        
        # --- NEW: Screenshot Capture Logic ---
        if step in target_steps:
            # We will save these temporarily in the results folder
            filename = f"results/{formation_type}/temp_step_{step}.png"
            env.fig.savefig(filename, bbox_inches='tight', dpi=150)
            saved_screenshots.append(filename)
            print(f"Captured screenshot at step {step}")
            
        time.sleep(0.01) # Slow down slightly for visual clarity

    print(f"Evaluation Finished. Total Swarm Reward: {total_reward:.2f}")

    # --- NEW: Image Stitching Logic ---
    if len(saved_screenshots) == 3:
        print("Stitching screenshots into a single progress image...")
        
        # Open all 3 images
        images = [Image.open(img_path) for img_path in saved_screenshots]
        
        # Calculate dimensions for the final combined image
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)

        # Create a new blank canvas
        combined_img = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))

        # Paste the images side-by-side
        x_offset = 0
        for img in images:
            combined_img.paste(img, (x_offset, 0))
            x_offset += img.size[0]

        # Save the final masterpiece
        final_path = f"results/{formation_type}/{formation_type}_progress_showcase.png"
        combined_img.save(final_path)
        print(f"SUCCESS! Progress showcase saved to: {final_path}")

        # Clean up the temporary individual files
        for img_path in saved_screenshots:
            os.remove(img_path)

if __name__ == "__main__":
    subprocess.run("cls", shell=True, check=True)
    
    # Evaluate all configurations found in the configs folder
    for config_file in os.listdir('configs'):
        if config_file.endswith('.json'):
            print(f"\n--- Evaluating {config_file} ---")
            config_path = os.path.join('configs', config_file)
            evaluate(config_path)