import os
from typing import Optional

import random
import torch
import numpy as np
from ..estimator import MaxQEstimator
from ..networks import QNetwork
from collections import namedtuple
from stable_baselines3.common.buffers import ReplayBuffer
from gymnasium.vector import SyncVectorEnv


Experience = namedtuple(
    "Experience",
    field_names=["obs", "actions", "next_obs", "terminated", "rewards", "info"],
)


class Agent:
    def __init__(
        self,
        envs: SyncVectorEnv,
        replay_buffer: Optional[ReplayBuffer] = None,
        local_network: Optional[QNetwork] = None,
        target_network: Optional[QNetwork] = None,
        estimator: Optional[MaxQEstimator] = None,
        batch_size: int = 128,
        gamma: float = 0.99,
        device: str = "cpu",
        model_path: Optional[str] = None,
    ):
        self.envs = envs
        self.device = device

        if local_network and target_network:
            self.local_network = local_network
            self.target_network = target_network
        elif model_path:
            self.load(model_path)
        else:
            raise AttributeError("Either Networks work Model Path need to be provided.")

        self.replay_buffer = replay_buffer
        self.estimator = estimator

        self.batch_size = batch_size
        self.gamma = gamma

        self.loss = None
        self.step = 0

    def act(self, obs: np.ndarray, epsilon: float) -> np.ndarray:
        if random.random() < epsilon:
            actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
        else:
            q_values = self.local_network(torch.Tensor(obs).to(self.device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        return actions

    def update(self, experience: Experience, update_local: bool, update_target: bool) -> None:
        if not self.replay_buffer or not self.estimator:
            raise AttributeError("Replay buffer and Estimator are required to update.")

        # Handle final observation in sync enviroments
        obs, actions, next_obs, terminateds, rewards, infos = experience
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(terminateds):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]

        self.replay_buffer.add(obs, real_next_obs, actions, rewards, terminateds, infos)

        batch = self.replay_buffer.sample(self.batch_size)

        if update_local:
            # Update local network
            preds, targets = self.estimator(batch, self.local_network, self.target_network, self.gamma)
            self.loss = self.local_network.update(preds, targets)

        if update_target:
            # Update target nettork
            self.target_network.update_from(self.local_network)

        self.step += 1

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, "checkpoint.pt")
        checkpoint = {
            "local_network_state_dict": self.local_network.state_dict(),
            "local_network_hparams": self.local_network.hparams,
            "target_network_state_dict": self.target_network.state_dict(),
            "target_network_hparams": self.target_network.hparams,
        }
        torch.save(checkpoint, filepath)

    def load(self, path):
        filepath = os.path.join(path, "checkpoint.pt")
        checkpoint = torch.load(filepath, map_location=torch.device(self.device))

        self.local_network = QNetwork(**checkpoint["local_network_hparams"]).to(self.device)
        self.local_network.load_state_dict(checkpoint["local_network_state_dict"])

        self.target_network = QNetwork(**checkpoint["target_network_hparams"]).to(self.device)
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
