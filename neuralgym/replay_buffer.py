from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class CustomReplayBuffer(ReplayBuffer):
    """
    A custom replay buffer that extends Stable Baselines3's ReplayBuffer.
    This implementation provides additional functionality while maintaining compatibility
    with Stable Baselines3's algorithms.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        """
        Initialize the replay buffer.

        Args:
            buffer_size: Max number of element in the buffer
            observation_space: Observation space
            action_space: Action space
            device: Device to put the buffer on
            n_envs: Number of parallel environments
            optimize_memory_usage: Enable a memory efficient variant
            handle_timeout_termination: Handle timeout termination (due to timelimit)
                separately and treat the task as infinite horizon task.
        """
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )

        # Additional custom attributes
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.current_episode_reward: float = 0.0
        self.current_episode_length: int = 0

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Add a new transition to the buffer.

        Args:
            obs: Current observation
            next_obs: Next observation
            action: Action taken
            reward: Reward received
            done: Whether the episode is done
            infos: Additional information
        """
        super().add(obs, next_obs, action, reward, done, infos)

        # Update episode statistics
        self.current_episode_reward += reward[0]
        self.current_episode_length += 1

        if done[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0.0
            self.current_episode_length = 0

    def get_episode_statistics(self) -> Dict[str, float]:
        """
        Get statistics about the episodes in the buffer.

        Returns:
            Dictionary containing mean and std of episode rewards and lengths
        """
        if not self.episode_rewards:
            return {
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "mean_length": 0.0,
                "std_length": 0.0,
            }

        return {
            "mean_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "mean_length": np.mean(self.episode_lengths),
            "std_length": np.std(self.episode_lengths),
        }

    def sample(
        self,
        batch_size: int,
        env: Optional[VecNormalize] = None,
    ) -> ReplayBufferSamples:
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size: Number of transitions to sample
            env: Optional VecNormalize environment to normalize observations

        Returns:
            ReplayBufferSamples containing the sampled transitions
        """
        return super().sample(batch_size, env)

    def save(self, path: str) -> None:
        """
        Save the replay buffer to disk.

        Args:
            path: Path to save the buffer
        """
        data = {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_observations": self.next_observations,
            "dones": self.dones,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
        }
        np.save(path, data)

    def load(self, path: str) -> None:
        """
        Load the replay buffer from disk.

        Args:
            path: Path to load the buffer from
        """
        data = np.load(path, allow_pickle=True).item()
        self.observations = data["observations"]
        self.actions = data["actions"]
        self.rewards = data["rewards"]
        self.next_observations = data["next_observations"]
        self.dones = data["dones"]
        self.episode_rewards = data["episode_rewards"]
        self.episode_lengths = data["episode_lengths"]
        self.pos = len(self.observations)
        self.full = self.pos >= self.buffer_size


# Example usage
if __name__ == "__main__":
    import gymnasium as gym

    # Create a simple environment
    env = gym.make("CartPole-v1")

    # Initialize the replay buffer
    buffer = CustomReplayBuffer(
        buffer_size=10000,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    # Collect some experience
    obs, _ = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.add(
            obs=np.array([obs]),
            next_obs=np.array([next_obs]),
            action=np.array([action]),
            reward=np.array([reward]),
            done=np.array([done]),
            infos=[{}],
        )

        obs = next_obs
        if done:
            obs, _ = env.reset()

    # Get episode statistics
    stats = buffer.get_episode_statistics()
    print("Episode Statistics:", stats)

    # Sample a batch
    batch = buffer.sample(batch_size=32)
    print("Batch shapes:")
    print(f"Observations: {batch.observations.shape}")
    print(f"Actions: {batch.actions.shape}")
    print(f"Rewards: {batch.rewards.shape}")
    print(f"Next observations: {batch.next_observations.shape}")
    print(f"Dones: {batch.dones.shape}")

    # Save and load the buffer
    buffer.save("replay_buffer.npy")
    new_buffer = CustomReplayBuffer(
        buffer_size=10000,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )
    new_buffer.load("replay_buffer.npy")
