from typing import Optional

import gymnasium as gym
from gymnasium.wrappers import (
    FrameStack,
    GrayScaleObservation,
    RecordEpisodeStatistics,
    ResizeObservation,
)
from gymnasium.vector import SyncVectorEnv
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


def get_atari_env(env_id: str, seed: int, idx: int = 0, human_render: bool = False, run_name: Optional[str] = None):
    if human_render and idx == 0:
        env = gym.make(env_id, render_mode="human")
    else:
        env = gym.make(env_id, render_mode="rgb_array")

    env = RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = ResizeObservation(env, (84, 84))
    env = GrayScaleObservation(env)
    env = FrameStack(env, 4)
    env.action_space.seed(seed)

    return env


def make_sync_envs(env_id: str, n_envs: int, seed: int, capture_video: bool = False, run_name: Optional[str] = None):
    return SyncVectorEnv([lambda: get_atari_env(env_id, seed + i, i, capture_video, run_name) for i in range(n_envs)])
