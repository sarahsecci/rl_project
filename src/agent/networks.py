"""
Neural network architectures (MLP and CNN) for DQN and RND agents.
Authors: Clara Schindler and Sarah Secci
Date: 09-08-25
Parts of this code were made with the help of Copilot
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron (MLP) for DQN and RND.
    The architecture is designed to be similar to the original DQN paper https://arxiv.org/pdf/1312.5602.
    """

    def __init__(self, input_size: int, output_size: int, hidden_dim: int = 64):
        """
        Initialize MLP with specified architecture.

        Parameters
        ----------
        input_size : int
            Size of the input layer (e.g., flattened observation size).
        output_size : int
            Size of the output layer (e.g., number of actions).
        hidden_dim : int
            Size of hidden layers.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_size).
        """
        return self.fc(x)


class CNN(nn.Module):
    """
    CNN designed to handle RGB images as input.
    The architecture is designed analogous to the original DQN paper https://arxiv.org/pdf/1312.5602.
    """

    def __init__(self, obs_shape: tuple, output_size: int, hidden_dim: int = 64):
        """
        Initialize CNN with specified architecture.

        Parameters
        ----------
        obs_shape : tuple
            Shape of the input observation (C, H, W) format.
        output_size : int
            Size of the output layer (e.g., number of actions).
        hidden_dim : int
            Base size for hidden layers.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Accepts (B, H, W, C) or (B, C, H, W) format.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_size).
        """
        # Accepts (B, H, W, C) or (B, C, H, W)
        if x.ndim == 4 and x.shape[1] != 3 and x.shape[-1] == 3:
            # Convert (B, H, W, C) to (B, C, H, W)
            x = x.permute(0, 3, 1, 2)
        return self.fc(self.conv(x))
