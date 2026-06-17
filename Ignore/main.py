import argparse
import time
import gym
import numpy as np
import glfw 
from stable_baselines3 import DQN, PPO
from Environment.custom_env import IndoorNavEnv
from Environment.rendering import HouseVisualizer

def visualize_agent(model_path, model_type='dqn', num_episodes=3, fps=5, step_delay=0.5):
    # Initialize environment and model
    env = IndoorNavEnv()
    env.target_sequence = ['living_room', 'kitchen', 'bathroom', 'bedroom']  # Custom order
    
    if model_type.lower() == 'dqn':
        model = DQN.load(model_path)
    elif model_type.lower() == 'ppo':
        model = PPO.load(model_path)
    else:
        raise ValueError("Model type must be 'dqn' or 'ppo'")

    # Initialize GLFW-based visualizer
    visualizer = HouseVisualizer(env)
    
    for episode in range(num_episodes):
        obs = env.reset()
        trajectory = []  # Initialize trajectory storage
        done = False
        total_reward = 0
        start_time = time.time()

        while not done and not glfw.window_should_close(visualizer.window):
            # Get agent action
            action, _ = model.predict(obs)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            total_reward += reward

            # Render using GLFW
            trajectory.append(env.agent_pos)
            visualizer.render(env.agent_pos, trajectory)
            time.sleep(step_delay)  # Add explicit delay after each step
            # GLFW event handling
            glfw.poll_events()
            
            # Control frame rate using time
            time.sleep(1/fps)

        # Early exit if window closed
        if glfw.window_should_close(visualizer.window):
            break
            
        # Episode summary
        duration = time.time() - start_time
        print(f"\nEpisode {episode+1} Results:")
        print(f"- Target Room: {env.target_room}")
        print(f"- Final Position: {env.agent_pos}")
        print(f"- Doors Passed: {info['doorways_passed']}")  # From step info
        print(f"- Total Reward: {total_reward:.1f}")
        print(f"- Duration: {duration:.1f}s")
        print(f"- Steps: {env.current_step}")
        print(f"- Collision: {'Yes' if info.get('collision', False) else 'No'}")
        print(f"- Target Reached: {'Yes' if env.in_target_room() else 'No'}")
    visualizer.close()

if __name__ == "__main__":
    # Argument parsing should be INSIDE this block
    parser = argparse.ArgumentParser(description='Visualize RL Agent Navigation')
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to trained model (PPO/DQN)')
    parser.add_argument('--model-type', type=str, choices=['dqn', 'ppo'], default='ppo',
                      help='Type of RL model (dqn/ppo)')
    parser.add_argument('--episodes', type=int, default=3,
                      help='Number of episodes to visualize')
    parser.add_argument('--fps', type=int, default=5,
                      help='Rendering frames per second (lower = slower)')
    parser.add_argument('--step-delay', type=float, default=0.3,
                      help='Additional delay after each step (seconds)')
    
    args = parser.parse_args()

    visualize_agent(
        model_path=args.model_path,
        model_type=args.model_type,
        num_episodes=args.episodes,
        fps=args.fps,
        step_delay=args.step_delay
    )