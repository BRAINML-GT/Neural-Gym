import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import pickle

class DopamineEnv(gym.Env):
    """
    An offline RL environment based on dopamine level data from mice.
    
    This environment uses pre-recorded syllable sequences and dopamine levels
    to create an environment for offline reinforcement learning.
    """
    
    def __init__(self, data_path=None, max_steps=100):
        """
        Initialize the Dopamine environment.
        
        Args:
            data_path: Path to the data file (default: None, will use default path)
            max_steps: Maximum number of steps in an episode
        """
        super(DopamineEnv, self).__init__()
        
        # Default data path
        if data_path is None:
            data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     "data/dopamine_level/data_by_mouse_id.npy")
        
        # Load data
        self.seqs, self.DAs, self.z_DAs, self.timestamps = self._load_data(data_path)
        
        # Process data to get all sequences and DAs
        self.all_seqs, self.all_DAs, self.all_raw_DAs = self._process_data()
        
        # Get the number of unique syllables
        self.n_syllables = np.max(self.all_seqs) + 1
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.n_syllables)
        self.observation_space = spaces.Discrete(self.n_syllables)
        
        # Environment properties
        self.max_steps = max_steps
        self.current_step = 0
        self.current_episode = 0
        self.current_state = None
        self.episode_transitions = []
        
        # Initialize state
        self.reset()
    
    def _load_data(self, data_path):
        """Load the data from the given path."""
        try:
            seqs, DAs, z_DAs, timestamps = np.load(data_path, allow_pickle=True)
            return seqs, DAs, z_DAs, timestamps
        except Exception as e:
            raise ValueError(f"Failed to load data from {data_path}: {e}")
    
    def _process_data(self):
        """Process the data to get all sequences and DAs across mice."""
        all_seqs = []
        all_DAs = []
        all_raw_DAs = []
        
        mouse_ids = self.seqs.keys()
        for mouse_id in mouse_ids:
            curr_seqs = self.seqs[mouse_id]
            curr_DAs = self.z_DAs[mouse_id]
            curr_raw_DAs = self.DAs[mouse_id]
            all_seqs.append(curr_seqs)
            all_DAs.append(curr_DAs)
            all_raw_DAs.append(curr_raw_DAs)
        
        all_seqs = np.concatenate(all_seqs, axis=0)
        all_DAs = np.concatenate(all_DAs, axis=0)
        all_raw_DAs = np.concatenate(all_raw_DAs, axis=0)
        
        return all_seqs, all_DAs, all_raw_DAs
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new episode.
        
        Returns:
            observation: The initial observation
            info: Additional information
        """
        # Select a random sequence from the dataset
        self.current_episode = np.random.randint(0, len(self.all_seqs))
        self.current_step = 0
        self.current_state = self.all_seqs[self.current_episode, self.current_step]
        self.episode_transitions = []
        
        info = {}
        return self.current_state, info
    
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
        if self.current_step >= self.max_steps - 1 or self.current_step >= len(self.all_seqs[self.current_episode]) - 2:
            return self.current_state, 0, True, True, {}
        
        # Get the actual next state and reward from the data
        self.current_step += 1
        next_state = self.all_seqs[self.current_episode, self.current_step]
        reward = self.all_DAs[self.current_episode, self.current_step]
        
        # Record the transition
        transition = {
            "state": self.current_state,
            "action": action,
            "next_state": next_state,
            "reward": reward,
            "real_next_state": next_state,
            "real_reward": reward
        }
        self.episode_transitions.append(transition)
        
        # Update current state
        self.current_state = next_state
        
        # Check if the episode is terminated
        terminated = self.current_step >= len(self.all_seqs[self.current_episode]) - 1
        truncated = self.current_step >= self.max_steps - 1
        
        info = {}
        return next_state, reward, terminated, truncated, info
    
    def build_replay_buffer(self, buffer_size=None):
        """
        Build a replay buffer from the dataset.
        
        Args:
            buffer_size: Maximum size of the buffer (default: None, uses all data)
            
        Returns:
            replay_buffer: A dictionary containing the replay buffer with only the first 10 syllables
        """
        # First count the maximum number of transitions we might collect
        max_transitions = 0
        for ep_idx in range(len(self.all_seqs)):
            max_transitions += len(self.all_seqs[ep_idx]) - 1
        
        # Determine provisional buffer size (might be adjusted later)
        if buffer_size is None or buffer_size > max_transitions:
            provisional_buffer_size = max_transitions
        else:
            provisional_buffer_size = buffer_size
        
        # Initialize arrays to store transitions
        states = np.zeros(provisional_buffer_size, dtype=np.int32)
        actions = np.zeros(provisional_buffer_size, dtype=np.int32)
        next_states = np.zeros(provisional_buffer_size, dtype=np.int32)
        rewards = np.zeros(provisional_buffer_size, dtype=np.float32)
        dones = np.zeros(provisional_buffer_size, dtype=np.bool_)
        
        # Fill the arrays with valid transitions
        idx = 0
        for ep_idx in range(len(self.all_seqs)):
            seq_len = len(self.all_seqs[ep_idx])
            for step_idx in range(seq_len - 1):
                if idx >= provisional_buffer_size:
                    break
                
                state = self.all_seqs[ep_idx, step_idx]
                next_state = self.all_seqs[ep_idx, step_idx + 1]
                
                # Only include transitions where both states are < 10 (first 10 syllables)
                if state < 10 and next_state < 10:
                    reward = self.all_DAs[ep_idx, step_idx]
                    done = step_idx == seq_len - 2
                    
                    # Assuming the action taken was the actual next syllable
                    action = next_state
                    
                    states[idx] = state
                    actions[idx] = action
                    next_states[idx] = next_state
                    rewards[idx] = reward
                    dones[idx] = done
                    
                    idx += 1
        
        # Create the final replay buffer with the correct size
        actual_buffer_size = idx
        replay_buffer = {
            "states": states[:actual_buffer_size],
            "actions": actions[:actual_buffer_size],
            "next_states": next_states[:actual_buffer_size],
            "rewards": rewards[:actual_buffer_size],
            "dones": dones[:actual_buffer_size]
        }
        
        return replay_buffer
    
    def save_replay_buffer(self, path="replay_buffer.pkl", buffer_size=None):
        """
        Save the replay buffer to a file.
        
        Args:
            path: Path to save the replay buffer
            buffer_size: Maximum size of the buffer
        """
        replay_buffer = self.build_replay_buffer(buffer_size)
        
        with open(path, 'wb') as f:
            pickle.dump(replay_buffer, f)
        
        print(f"Replay buffer saved to {path}")
    
    def get_transition_stats(self):
        """
        Get statistics about transitions between syllables.
        
        Returns:
            A dictionary with statistics about transitions
        """
        transitions = {}
        transition_counts = {}
        
        # Count transitions and collect rewards
        for ep_idx in range(len(self.all_seqs)):
            seq_len = len(self.all_seqs[ep_idx])
            for step_idx in range(seq_len - 1):
                curr_state = self.all_seqs[ep_idx, step_idx]
                next_state = self.all_seqs[ep_idx, step_idx + 1]
                reward = self.all_DAs[ep_idx, step_idx]
                
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
                'count': transition_counts[key],
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'min_reward': np.min(rewards),
                'max_reward': np.max(rewards)
            }
        
        return stats
    
    def render(self):
        """Render the environment."""
        pass  # Not implemented for this environment

# Example usage
if __name__ == "__main__":
    env = DopamineEnv()
    print(f"Loaded environment with {env.n_syllables} syllables")
    print(f"Data shape: {env.all_seqs.shape}, {env.all_DAs.shape}")
    
    # Build and save replay buffer
    env.save_replay_buffer("dopamine_replay_buffer.pkl")
    
    # Get transition statistics
    stats = env.get_transition_stats()
    print(f"Computed statistics for {len(stats)} different transitions")
    
    # Example of running a few episodes
    for episode in range(3):
        state, _ = env.reset()
        total_reward = 0
        step = 0
        
        while True:
            # In offline RL, we don't actually select actions
            # but we can simulate by using the next state as the action
            action = np.random.randint(0, env.n_syllables)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            total_reward += reward
            step += 1
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode+1} finished with {step} steps and reward {total_reward}")