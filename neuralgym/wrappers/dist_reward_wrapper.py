from __future__ import annotations

from typing import Callable, SupportsFloat

import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.error import InvalidBound

__all__ = ["StdGaussianRewardWrapper"]


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
        return reward + np.random.normal(0, 1)
