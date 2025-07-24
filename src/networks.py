"""
Convolutional networks for DQN and RND.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron (MLP) for DQN and RND.
    The architecture is designed to be similar to the original DQN paper https://arxiv.org/pdf/1312.5602.
    """

    def __init__(self, input_size, output_size, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size),
        )

    def forward(self, x):
        return self.fc(x)


class CNN(nn.Module):
    """
    CNN designed to handle RGB images as input.
    The architecture is designed analogous to the original DQN paper https://arxiv.org/pdf/1312.5602.
    """

    def __init__(self, obs_shape, output_size, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, (2 * hidden_dim), kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Dynamically compute the flattened size after conv layers
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            n_flat = self.conv(dummy).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(n_flat, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_size)
        )

    def forward(self, x):
        # Accepts (B, H, W, C) or (B, C, H, W)
        if x.ndim == 4 and x.shape[1] != 3 and x.shape[-1] == 3:
            # Convert (B, H, W, C) to (B, C, H, W)
            x = x.permute(0, 3, 1, 2)
        return self.fc(self.conv(x))
