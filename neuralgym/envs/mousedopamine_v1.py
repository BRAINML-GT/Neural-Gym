import os
import pickle
from typing import Optional, Any
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.save_util import load_from_pkl


class MouseDopamineEnv(gym.Env):
    """
    An offline RL environment based on dopamine level data from mice.

    This environment uses pre-processed replay buffer data containing
    syllable sequences and dopamine levels for offline reinforcement learning.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        max_syllables: Optional[int] = None,
        mouse_id: Optional[str] = None,
        trial_id: Optional[int] = None,
        use_trial_id: Optional[bool] = False,
    ):
        """
        Initialize the Mouse Dopamine environment.

        Args:
            data_path: Path to the replay buffer file (default: None, will use default path)
            max_syllables: Maximum number of syllables to consider (default: None, uses all)
            mouse_id: Specific mouse ID to use (default: None, uses all mice)
            trial_id: Specific trial ID to use (default: None, uses all trials)
            use_trial_id: Whether to use trial ID (default: False)
        """
        super(MouseDopamineEnv, self).__init__()

        # Default data path
        if data_path is None:
            data_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "data/dopamine_level/mouse_rb.pkl",
            )

        # Load replay buffer
        self.replay_buffer = load_from_pkl(data_path)

        # Get the number of unique syllables from the observation space
        self.n_syllables = self.replay_buffer.observation_space.n

        # Set the maximum number of syllables, if provided
        if max_syllables is not None:
            self.max_syllables = min(max_syllables, self.n_syllables)
        else:
            self.max_syllables = self.n_syllables

        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.max_syllables)
        self.observation_space = spaces.Discrete(self.max_syllables)

        # Environment properties
        self.current_step = 0
        self.current_episode = 0
        self.current_state = None
        self.episode_transitions = []

        # Initialize state
        self.reset()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """
        Reset the environment to start a new episode.

        Returns:
            observation: The initial observation
            info: Additional information
        """
        # Select a random transition from the replay buffer
        self.current_step = np.random.randint(0, len(self.replay_buffer.observations))
        self.current_state, _, _, _, _ = self.replay_buffer.sample(1)
        self.episode_transitions = []

        # Set the seed if provided
        if seed is not None:
            np.random.seed(seed)

        info = {}
        return self.current_state.item(), info

    def step(self, action):
        """
        Take a step in the environment with the given action.

        Args:
            action: The action to take

        Returns:
            next_state: The next state
            reward: The reward received
            terminated: Whether the episode is terminated
            truncated: Whether the episode is truncated
            info: Additional information
        """
        # Get the next state and reward from the replay buffer
        next_state = self.replay_buffer.next_observations[self.current_step]
        reward = self.replay_buffer.rewards[self.current_step]
        done = self.replay_buffer.dones[self.current_step]

        # Record the transition
        transition = {
            "state": self.current_state,
            "action": action,
            "next_state": next_state,
            "reward": reward,
            "real_next_state": next_state,
            "real_reward": reward,
        }
        self.episode_transitions.append(transition)

        # Update current state and step
        self.current_state = next_state
        self.current_step += 1

        # Check if we've reached the end of the buffer
        if self.current_step >= len(self.replay_buffer.observations):
            done = True

        info = {}
        return next_state, reward, done, False, info

    def get_transition_stats(self):
        """
        Get statistics about transitions between syllables.

        Returns:
            A dictionary with statistics about transitions
        """
        transitions = {}
        transition_counts = {}

        # Count transitions and collect rewards
        for i in range(len(self.replay_buffer.observations)):
            curr_state = self.replay_buffer.observations[i]
            next_state = self.replay_buffer.next_observations[i]
            reward = self.replay_buffer.rewards[i]

            key = (curr_state, next_state)
            if key not in transitions:
                transitions[key] = []
                transition_counts[key] = 0

            transitions[key].append(reward)
            transition_counts[key] += 1

        # Calculate stats
        stats = {}
        for key, rewards in transitions.items():
            stats[key] = {
                "count": transition_counts[key],
                "mean_reward": np.mean(rewards),
                "std_reward": np.std(rewards),
                "min_reward": np.min(rewards),
                "max_reward": np.max(rewards),
            }

        return stats

    def render(self):
        """Render the environment."""
        pass  # Not implemented for this environment
