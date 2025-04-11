from neuralgym.envs.mousedopamine_v1 import MouseDopamineEnv
import gymnasium as gym

gym.register(
    id="MouseDopamineEnv-v1",
    entry_point="neuralgym.envs.mousedopamine_v1:MouseDopamineEnv",
)

__all__ = ["MouseDopamineEnv"]
