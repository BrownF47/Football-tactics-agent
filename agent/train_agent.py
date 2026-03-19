import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from environment.gym_environment import FootballEnv

# Load data
shots_df = pd.read_parquet('data/shots.parquet')

# Create environment
env = FootballEnv(shots_df, repositioning_cost=0.01)
env = Monitor(env)  # Wraps env to log episode rewards

# Train PPO agent
model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    tensorboard_log='./runs/'
)

print("Training agent...")
model.learn(total_timesteps=200_000)
model.save('agent/ppo_football')
print("Agent saved!")

# Evaluate against random baseline
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym


# Evaluate trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000)
print(f"\nPPO Agent:        {mean_reward:.4f} +/- {std_reward:.4f} mean xG")

# Evaluate random baseline
class RandomAgent:
    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        return [env.action_space.sample()], None

random_mean, random_std = evaluate_policy(RandomAgent(), env, n_eval_episodes=1000)
print(f"Random Baseline:  {random_mean:.4f} +/- {random_std:.4f} mean xG")

# Evaluate always-shoot baseline
class AlwaysShootAgent:
    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        return [4], None

shoot_mean, shoot_std = evaluate_policy(AlwaysShootAgent(), env, n_eval_episodes=1000)
print(f"Always Shoot:     {shoot_mean:.4f} +/- {shoot_std:.4f} mean xG")

print(f"\nPPO improvement over random: {((mean_reward - random_mean) / abs(random_mean)) * 100:.1f}%")
print(f"PPO improvement over always shoot: {((mean_reward - shoot_mean) / abs(shoot_mean)) * 100:.1f}%")