import os
import time
import numpy as np
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure, HParam
from Environment.custom_env import IndoorNavEnv
import matplotlib.pyplot as plt

class MetricTracker(BaseCallback):
    def __init__(self, verbose=0):
        super(MetricTracker, self).__init__(verbose)
        self.episode_metrics = {
            'rewards': [],
            'steps': [],
            'success': [],
            'collisions': []
        }
        self.start_time = time.time()
        self.convergence_step = None

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # Calculate metrics for completed episodes
        ep_info = self.model.ep_info_buffer
        for info in ep_info:
            self.episode_metrics['rewards'].append(info['r'])
            self.episode_metrics['steps'].append(info['l'])
            self.episode_metrics['success'].append(int(not info.get('collision', False) and info.get('done', False)))
            self.episode_metrics['collisions'].append(int(info.get('collision', False)))

            # Check convergence criteria (success rate > 80% for 10 consecutive episodes)
            if len(self.episode_metrics['success']) >= 10:
                last_10 = np.mean(self.episode_metrics['success'][-10:])
                if last_10 >= 0.8 and self.convergence_step is None:
                    self.convergence_step = self.num_timesteps

        # Log metrics to TensorBoard
        if len(ep_info) > 0:
            self.logger.record('metrics/avg_reward', np.mean(self.episode_metrics['rewards'][-10:]))
            self.logger.record('metrics/avg_steps', np.mean(self.episode_metrics['steps'][-10:]))
            self.logger.record('metrics/success_rate', np.mean(self.episode_metrics['success'][-10:]))
            self.logger.record('metrics/collision_rate', np.mean(self.episode_metrics['collisions'][-10:]))
            
            if self.convergence_step:
                self.logger.record('metrics/convergence_time', (time.time() - self.start_time)/60)  # Minutes
                self.logger.record('metrics/convergence_step', self.convergence_step)

        # Plot training progress
        fig = self._create_metrics_plot()
        self.logger.record('training_progress', Figure(fig, close=True), exclude=('stdout', 'log', 'json', 'csv'))

    def _create_metrics_plot(self):
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        
        # Success Rate
        axs[0,0].plot(np.convolve(self.episode_metrics['success'], np.ones(10)/10, mode='valid'))
        axs[0,0].set_title('Success Rate (10-episode avg)')
        
        # Collision Rate
        axs[0,1].plot(np.convolve(self.episode_metrics['collisions'], np.ones(10)/10, mode='valid'))
        axs[0,1].set_title('Collision Rate (10-episode avg)')
        
        # Average Steps
        axs[1,0].plot(np.convolve(self.episode_metrics['steps'], np.ones(10)/10, mode='valid'))
        axs[1,0].set_title('Average Steps (10-episode avg)')
        
        # Average Reward
        axs[1,1].plot(np.convolve(self.episode_metrics['rewards'], np.ones(10)/10, mode='valid'))
        axs[1,1].set_title('Average Reward (10-episode avg)')
        
        plt.tight_layout()
        return fig

# Training with Enhanced Metric Tracking
def train_dqn():
    env = DummyVecEnv([lambda: IndoorNavEnv()])
    
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./dqn_metrics/",
        learning_rate=0.0003,
        buffer_size=100000,
        batch_size=256,
        gamma=0.995,
        exploration_fraction=0.15,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        policy_kwargs={'net_arch': [512, 256]}
    )
    
    metric_callback = MetricTracker()
    
    model.learn(
        total_timesteps=200000,
        callback=metric_callback,
        tb_log_name="metric_tracking"
    )
    
    # Save final metrics
    np.savez("./training_metrics.npz", **metric_callback.episode_metrics)
    model.save("./dqn_metric_trained")
    
    return model

# Post-Training Evaluation
def evaluate_model(model_path, num_episodes=100):
    model = DQN.load(model_path)
    env = IndoorNavEnv()
    
    metrics = {
        'rewards': [],
        'steps': [],
        'success': [],
        'collisions': []
    }
    
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_steps = 0
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            
        metrics['rewards'].append(ep_reward)
        metrics['steps'].append(ep_steps)
        metrics['success'].append(int(not info['collision'] and done))
        metrics['collisions'].append(int(info['collision']))
    
    print("\nFinal Evaluation Metrics:")
    print(f"- Success Rate: {np.mean(metrics['success']):.2%}")
    print(f"- Collision Rate: {np.mean(metrics['collisions']):.2%}")
    print(f"- Avg Steps per Episode: {np.mean(metrics['steps']):.1f}")
    print(f"- Avg Reward per Episode: {np.mean(metrics['rewards']):.1f}")

# Run Training and Evaluation
if __name__ == "__main__":
    trained_model = train_dqn()
    evaluate_model("./dqn_metric_trained")