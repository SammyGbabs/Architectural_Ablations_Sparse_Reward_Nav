import os
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from Environment.custom_env import IndoorNavEnv  # Your custom environment

# Configuration
SEED = 42
LOG_DIR = "./dqn_logs/"
MODEL_DIR = "./dqn_models/"
TENSORBOARD_LOG = "./dqn_tensorboard/"

# Create environment
env = DummyVecEnv([lambda: IndoorNavEnv()])

# Test environment reset observation shape
obs = env.reset()
print("Initial Observation Shape:", obs.shape)

# Hyperparameters (optimized for indoor navigation)
params = {
    'learning_rate': 0.0003,
    'buffer_size': 100000,
    'batch_size': 256,
    'gamma': 0.99,
    'target_update_interval': 1000,
    'train_freq': 4,
    'gradient_steps': 1,
    'exploration_fraction': 0.15,
    'exploration_initial_eps': 1.0,
    'exploration_final_eps': 0.02,
    'policy_kwargs': {
        'net_arch': [512, 256]  # Larger network for complex navigation
    }
}

# Initialize model
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=TENSORBOARD_LOG,
    **params
)

# Callbacks
stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=5)
eval_callback = EvalCallback(
    env,
    best_model_save_path=MODEL_DIR,
    log_path=LOG_DIR,
    eval_freq=10000,
    callback_after_eval=stop_callback
)

# Training
total_timesteps = 200000
model.learn(
    total_timesteps=total_timesteps,
    callback=eval_callback,
    tb_log_name="dqn_run",
    reset_num_timesteps=True
)

# Save final model
model.save(os.path.join(MODEL_DIR, "dqn_final_model"))

# Test the trained model
def test_model(model_path, num_episodes=10):
    model = DQN.load(model_path)
    env = IndoorNavEnv()
    
    success = 0
    collision = 0
    total_steps = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        steps = 0
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            steps += 1
            
            if done:
                if info['collision']:
                    collision += 1
                elif info.get('timeout', False):
                    pass
                else:
                    success += 1
                total_steps.append(steps)
        
    print(f"Success Rate: {success/num_episodes:.2f}")
    print(f"Collision Rate: {collision/num_episodes:.2f}")
    print(f"Avg Steps: {sum(total_steps)/len(total_steps):.2f}")

# Evaluate best model
test_model(os.path.join(MODEL_DIR, "best_model"))