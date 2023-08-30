from typing import Tuple

import torch
from torch import Tensor
from ..networks import QNetwork


class MaxQEstimator:
    def __init__(self):
        """Creates a Q estimator

        Args:
            local_network (Module): the local, more frequently updated brain
            target_network (Module): the target, more stable brain
            gamma (float): Gamma parameter or discount factor
        """

    def __call__(
        self, sample, local_network: QNetwork, target_network: QNetwork, gamma: float
    ) -> Tuple[Tensor, Tensor]:
        obs, actions, next_obs, terminated, rewards = sample

        with torch.no_grad():
            # Implement DQN
            max_vals = target_network(next_obs).max(dim=1).values
            target = rewards.squeeze() + gamma * max_vals * (1 - terminated.squeeze())

        pred_values = local_network(obs)
        pred = pred_values.gather(1, actions).squeeze()

        return pred, target
