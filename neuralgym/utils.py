from typing import Dict, Optional, Tuple

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

from .dopamine_level import DopamineEnv


def create_dopamine_replay_buffer(
    env: DopamineEnv,
    buffer_size: Optional[int] = None,
) -> ReplayBuffer:
    """
    Create a Stable Baselines3 replay buffer from the dopamine level environment data.

    Args:
        env: The DopamineEnv instance containing the data
        buffer_size: Optional maximum size of the buffer. If None, uses all available data.

    Returns:
        ReplayBuffer: A Stable Baselines3 replay buffer containing the transitions
    """
    # Build the raw replay buffer from the environment
    raw_buffer = env.build_replay_buffer(buffer_size)

    # Create SB3 replay buffer
    sb3_buffer = ReplayBuffer(
        buffer_size=len(raw_buffer["states"]),
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    # Convert the data to the format expected by SB3
    observations = raw_buffer["states"].reshape(-1, 1)  # Add batch dimension
    next_observations = raw_buffer["next_states"].reshape(-1, 1)
    actions = raw_buffer["actions"].reshape(-1, 1)
    rewards = raw_buffer["rewards"].reshape(-1, 1)
    dones = raw_buffer["dones"].reshape(-1, 1)

    # Add all transitions to the buffer
    for i in range(len(observations)):
        sb3_buffer.add(
            obs=observations[i],
            next_obs=next_observations[i],
            action=actions[i],
            reward=rewards[i],
            done=dones[i],
            infos=[{}],
        )

    return sb3_buffer


def save_dopamine_replay_buffer(
    env: DopamineEnv,
    path: str,
    buffer_size: Optional[int] = None,
) -> None:
    """
    Create and save a Stable Baselines3 replay buffer from the dopamine level environment data.

    Args:
        env: The DopamineEnv instance containing the data
        path: Path to save the replay buffer
        buffer_size: Optional maximum size of the buffer. If None, uses all available data.
    """
    buffer = create_dopamine_replay_buffer(env, buffer_size)
    buffer.save(path)


def load_dopamine_replay_buffer(
    env: DopamineEnv,
    path: str,
) -> ReplayBuffer:
    """
    Load a previously saved Stable Baselines3 replay buffer for the dopamine environment.

    Args:
        env: The DopamineEnv instance
        path: Path to load the replay buffer from

    Returns:
        ReplayBuffer: The loaded replay buffer
    """
    buffer = ReplayBuffer(
        buffer_size=1,  # Temporary size, will be updated when loading
        observation_space=env.observation_space,
        action_space=env.action_space,
    )
    buffer.load(path)
    return buffer


# Example usage
if __name__ == "__main__":
    # Create the environment
    env = DopamineEnv()

    # Create and save a replay buffer
    save_dopamine_replay_buffer(env, "dopamine_sb3_buffer.npz")

    # Load the replay buffer
    loaded_buffer = load_dopamine_replay_buffer(env, "dopamine_sb3_buffer.npz")

    # Sample a batch
    batch = loaded_buffer.sample(batch_size=32)
    print("Batch shapes:")
    print(f"Observations: {batch.observations.shape}")
    print(f"Actions: {batch.actions.shape}")
    print(f"Rewards: {batch.rewards.shape}")
    print(f"Next observations: {batch.next_observations.shape}")
    print(f"Dones: {batch.dones.shape}")
