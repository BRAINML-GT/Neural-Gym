import numpy as np
import pickle
import gymnasium as gym

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl

if __name__ == "__main__":
    # create the environment
    env = gym.make("MouseDopamineEnv-v1")

    # create the replay buffer
    rb = ReplayBuffer(
        buffer_size=1000000,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )
    # Load the replay buffer
    replay_buffer = load_from_pkl("dopamine_sb3_buffer.pkl")

    # Print the replay buffer
    print(replay_buffer)
