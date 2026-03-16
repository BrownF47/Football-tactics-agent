import numpy as np
import gymnasium as gym

class FootballEnv():

    def __init__(self, shots_df):
        super().__init__()
        self.shots_df = shots_df
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(8,), dtype=np.float32)


    def reset(self, seed=None):
        self.zone = np.random.randint(0, 15)
        self.time = np.random.randint(0, 3)
        self.score_diff = 0
        return self._get_obs(), {}

    def step(self, action):
        reward = self._compute_reward(action)
        done = action == 4  # Shoot ends episode
        self._transition(action)
        return self._get_obs(), reward, done, False, {}


    def _compute_reward(self, action):
        if action == 4:  # Shoot
            # Sample xG from real shots in this zone
            zone_shots = self.shots_df[self.shots_df['zone'] == self.zone]
            return float(zone_shots['xG'].mean()) if len(zone_shots) > 0 else 0.0
        return 0.0


#from stable_baselines3.common.env_checker import check_env
#env = FootballEnv(shots_df)
#check_env(env)  # Will raise errors if your env is malformed
