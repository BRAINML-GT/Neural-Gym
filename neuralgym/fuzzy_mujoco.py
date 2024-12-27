import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv


class FuzzyMujocoEnv(MujocoEnv):

    def __init__(self, mean_vec: np.ndarray, variance_vec: np.ndarray, **kwargs):
        super(FuzzyMujocoEnv, self).__init__(**kwargs)
        self.act_dim = mean_vec.shape[0]

        self.mean_vec = mean_vec
        self.variance_vec = variance_vec

    def step(self, action):
        obs, reward, done, info = super(FuzzyMujocoEnv, self).step(action)
        mu = abs(action) @ self.mean_vec / self.act_dim
        sigma = abs(action) @ self.variance_vec
        # cap the mean and variance
        mu = np.clip(mu, -1, 1)
        sigma = np.clip(sigma, 0, 5)

        reward += np.random.normal(mu, sigma)
        return obs, reward, done, info
