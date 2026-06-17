import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from Environment.custom_env import IndoorNavEnv
from stable_baselines3.common.env_checker import check_env


# Configuration
SEED = 42
LOG_DIR = "./ppo_logs/"
MODEL_DIR = "./ppo_models/"
TENSORBOARD_LOG = "./ppo_tensorboard/"


# Create environment
# The error was caused because the action space was being interpreted as the observation space.
# To fix this, ensure you are using the correct policy type that corresponds to your action space.
# Since your action space is `Discrete(5)`, you should use a policy that handles discrete actions, such as 'MlpPolicy'.
env = IndoorNavEnv() # Initialize a single environment 
# Check the environment 
check_env(env, warn=True)  
# Wrap the environment with the appropriate wrappers
env = Monitor(env) # Wrap the environment for monitoring if needed
env = DummyVecEnv([lambda: env]) # Wrap to make it compatible with Stable-Baselines3 
# You can also wrap with VecCheckNan if you have NaN values in your observations
env = VecCheckNan(env, raise_exception=True) # Detects and stops on NaN observations

print(f"Action Space: {env.action_space}")
print(f"Observation Space: {env.observation_space}")

# Hyperparameters optimized for navigation tasks
params = {
    'learning_rate': 3e-4,
    'n_steps': 1024,
    'batch_size': 256,
    'n_epochs': 7,
    'gamma': 0.995,
    'gae_lambda': 0.92,
    'clip_range': 0.2,
    'ent_coef': 0.01,            
    'policy_kwargs': dict(
        net_arch=dict(pi=[512, 256, 128], vf=[256, 128]), # Changed to dictionary
        activation_fn=nn.LeakyReLU,
        ortho_init=True
    )
}

# Initialize model
# Use 'MlpPolicy' which is suitable for discrete action spaces.
# The original code has an issue with how it's initialized. Removing the action space from supported_action_spaces to allow it to automatically infer it.
model = PPO(
    "MlpPolicy",  # Changed back to MlpPolicy
    env,
    verbose=1,
    tensorboard_log=TENSORBOARD_LOG,
    **params,
    # Removing 'supported_action_spaces' as it's no longer a valid argument for PPO
)

# Callbacks
stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=5)
eval_callback = EvalCallback(
    env,
    best_model_save_path=MODEL_DIR,
    log_path=LOG_DIR,
    eval_freq=20000,
    callback_after_eval=stop_callback
)

# Training
total_timesteps = 500000  
model.learn(
    total_timesteps=total_timesteps,
    callback=eval_callback,
    tb_log_name="ppo_run",
    reset_num_timesteps=True
)

# Save final model
model.save(os.path.join(MODEL_DIR, "ppo_final_model"))

# Evaluation function (No success/collision rate)
def evaluate_ppo_model(model_path, num_episodes=100):
    model = PPO.load(model_path)
    # Create a new environment for evaluation and wrap it 
    eval_env = IndoorNavEnv()
    eval_env = Monitor(eval_env)  # If you need to monitor the evaluation environment
    eval_env = DummyVecEnv([lambda: eval_env])
    
    rewards_per_episode = []
    steps_per_episode = []

    for _ in range(num_episodes):
        obs = eval_env.reset() # Get initial observation using VecEnv API
        done = False
        ep_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True) # Assuming you want deterministic actions
            obs, reward, done, _ = eval_env.step(action) # Use VecEnv API for step
            ep_reward += reward[0] # Access reward from the list
            steps += 1
            
        rewards_per_episode.append(ep_reward)
        steps_per_episode.append(steps)
    
    avg_reward = np.mean(rewards_per_episode)
    avg_steps = np.mean(steps_per_episode)

    print("\nPPO Evaluation Results:")
    print(f"Avg Reward: {avg_reward:.1f}")
    print(f"Avg Steps per Episode: {avg_steps:.1f}")

    return rewards_per_episode, steps_per_episode # Return steps_per_episode


# Evaluate best model
rewards, steps_per_episode = evaluate_ppo_model(os.path.join(MODEL_DIR, "best_model"))