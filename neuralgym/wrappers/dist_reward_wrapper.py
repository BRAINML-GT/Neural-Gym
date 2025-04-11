from __future__ import annotations

from typing import Callable, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.error import InvalidBound


class StdGaussianRewardWrapper(
    gym.RewardWrapper[ObsType, ActType], gym.utils.RecordConstructorArgs
):
    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
    ):
        """Initialize TransformReward wrapper.

        Args:
            env (Env): The environment to wrap
            func: (Callable): The function to apply to reward
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        """Apply function to reward.

        Args:
            reward (Union[float, int, np.ndarray]): environment's reward
        """
        # add a gaussian noise to the reward
        return float(reward) + np.random.normal(0, 1)


class RandomGaussianRewardWrapper(
    gym.Wrapper[ObsType, ActType, ObsType, ActType], gym.utils.RecordConstructorArgs
):
    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        mean_vec: np.ndarray,
        variance_vec: np.ndarray,
    ):
        """Initialize TransformReward wrapper.

        Args:
            env (Env): The environment to wrap
            func: (Callable): The function to apply to reward
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.RewardWrapper.__init__(self, env)

        if mean_vec.shape != variance_vec.shape:
            raise InvalidBound("mean_vec and variance_vec must have the same shape")

        self.act_dim = mean_vec.shape[0]

        self.mean_vec = mean_vec
        self.variance_vec = variance_vec

    # modify step function so that we create two vectors mean_vec and variance_vec
    # then we compute the gaussian mean as a @ mean_vec and the gaussian variance as a @ variance_vec. Cap the mean and variance to be within a certain range
    # then sample from the gaussian distribution and add it to the reward
    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, dict]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        mu = action @ self.mean_vec
        sigma = abs(action) @ self.variance_vec
        # cap the mean and variance
        mu = np.clip(mu, -5, 5)
        sigma = np.clip(sigma, 1e-3, 10)
        # print(f"mu: {mu}, sigma: {sigma}")
        reward += np.random.normal(mu, sigma)
        return observation, reward, terminated, truncated, info

    def step_true_reward(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, dict]:
        observation, reward, terminated, truncated, info = self.env.step(action)

        return observation, reward, terminated, truncated, info

    @property
    def unwrapped(self) -> gym.Env[ObsType, ActType]:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self

    def get_reward_dist(self, action: ActType) -> tuple[float, float]:
        mu = action @ self.mean_vec
        sigma = abs(action) @ self.variance_vec
        return mu, sigma
