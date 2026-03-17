import numpy as np
import gymnasium as gym
import pandas as pd
import joblib
from stable_baselines3.common.env_checker import check_env

class FootballEnv(gym.Env):
    def __init__(self, shots_df, repositioning_cost=0.01):
        super().__init__()
        self.shots_df = shots_df
        self.xg_model = joblib.load('find_xG/xg_model.pkl')
        self.repositioning_cost = repositioning_cost

        # Continuous x,y state — no zones needed anymore
        self.x = 0.0
        self.y = 0.0

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(4,), dtype=np.float32)
            # (x, y, time, score_diff)

    def _get_xg(self):
        """Calculate xG from current position using your model"""
        distance = ((1 - self.x)**2 + (self.y - 0.5)**2)**0.5
        angle = np.arccos((1 - self.x) / (distance + 1e-6))  # 1e-6 avoids division by zero
        features = pd.DataFrame([[distance, angle]], 
                                  columns=['distance_from_goal_center', 'angle'])
        return float(self.xg_model.predict_proba(features)[:, 1])

    def _compute_reward(self, action):
        if action == 4:  # Shoot
            return self._get_xg()
        else:
            return -self.repositioning_cost

    def _get_obs(self):
        return np.array([
            self.x,
            self.y,
            self.time / 2.0,
            (self.score_diff + 2) / 4.0
        ], dtype=np.float32)

    def reset(self, seed=None):
        # Start from a random real shot position
        sample = self.shots_df.sample(1).iloc[0]
        self.x = float(sample['X'])
        self.y = float(sample['Y'])
        self.time = np.random.randint(0, 3)
        self.score_diff = 0
        return self._get_obs(), {}

    def _transition(self, action):
        step = 0.05  # How far the agent moves per action
        if action == 0:   # Move wide
            self.y = np.clip(self.y + step, 0, 1)
        elif action == 1: # Move central
            self.y = self.y + step * (0.5 - self.y)  # Pull towards centre
        elif action == 2: # Move closer
            self.x = np.clip(self.x + step, 0, 1)
        elif action == 3: # Hold position
            pass

    def step(self, action):
        reward = self._compute_reward(action)
        done = bool(action == 4)
        if not done:
            self._transition(action)
        return self._get_obs(), reward, done, False, {}



shots_df = pd.read_parquet('data/shots.parquet')
env = FootballEnv(shots_df, repositioning_cost=0.01)

check_env(env)  # Prints warnings/errors if something is wrong
print("Environment passed all checks!")