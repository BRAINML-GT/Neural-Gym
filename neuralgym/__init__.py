__version__ = "0.1.0"
import os
from neuralgym import envs, wrappers
from neuralgym.replay_buffer import CustomReplayBuffer
from neuralgym.utils import (
    create_dopamine_replay_buffer,
    save_dopamine_replay_buffer,
    load_dopamine_replay_buffer,
)

# Assign the path to the data path using the script path
NEURALGYM_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data"
)


__all__ = [
    "CustomReplayBuffer",
    "create_dopamine_replay_buffer",
    "save_dopamine_replay_buffer",
    "load_dopamine_replay_buffer",
    "envs",
    "wrappers",
]
