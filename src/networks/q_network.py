import torch
from torch import Tensor, nn


class QNetwork(nn.Module):
    def __init__(self, lr_rate: float = 1e-4, tau: float = 1, n_frames: int = 4, n_actions: int = 6):
        super().__init__()
        self.hparams = {
            "lr_rate": lr_rate,
            "tau": tau,
            "n_frames": n_frames,
            "n_actions": n_actions,
        }
        self.lr_rate = lr_rate
        self.tau = tau
        self.model = self.__build_model(n_frames, n_actions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_rate)

    def __build_model(self, n_frames: int, n_actions: int):
        conv = nn.Sequential(
            nn.Conv2d(in_channels=n_frames, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # A fully connected layer takes the flattened
        # frame from third convolution layer, and outputs
        # 512 features. And the head gives the action values
        q_values = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 64, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=n_actions),
        )

        return nn.Sequential(conv, q_values)

    def forward(self, obs: Tensor):
        return self.model(obs / 255.0)

    def update(self, pred, target) -> Tensor:
        loss = nn.MSELoss()

        loss_val = loss(pred, target)
        self.optimizer.zero_grad()
        loss_val.backward()
        self.optimizer.step()

        return loss_val

    def update_from(self, update_network: "QNetwork"):
        for old_network_param, update_network_param in zip(self.model.parameters(), update_network.parameters()):
            old_network_param.data.copy_(
                self.tau * update_network_param.data + (1.0 - self.tau) * old_network_param.data
            )
