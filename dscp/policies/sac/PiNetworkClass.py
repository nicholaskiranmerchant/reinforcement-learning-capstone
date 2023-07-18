# Core imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import ipdb

# Typing imports
from typing import Tuple

class PiNetwork(nn.Module):
    def __init__(self, 
            state_dim: int,
            action_dim: int,
            action_upper_bounds: np.ndarray,
            ):
        super().__init__()

        assert action_upper_bounds.shape == (action_dim,)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_upper_bounds = torch.tensor(action_upper_bounds, dtype=torch.float32)

        # Tuneable variables
        self.l1_dim : int = 256
        self.l2_dim : int = 256

        # Worth cross checking with other SACs
        self.log_std_min : int = -20
        self.log_std_max : int = 2

        ## KIKI YOU STOPPED HERE

        # Build the Gaussian network
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.l1_dim),
            nn.ReLU(),
            nn.Linear(self.l1_dim, self.l2_dim),
            nn.ReLU(),
            )

        self.mu_layer = nn.Linear(self.l2_dim, output_dim)
        self.log_std_layer = nn.Linear(self.l2_dim, output_dim)

    def forward(self, input_: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        mlp_out = self.mlp(input_)
        mu = self.mu_layer(mlp_out)
        log_std = self.log_std_layer(mlp_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi = Normal(mu, std)
        output = pi.rsample()

        # Compute log probability, for entropy loss (I don't understand this)
        output_log_prob = pi.log_prob(output).sum(axis = -1)
        output_log_prob -= (2 * (np.log(2) - output - F.softplus(-2 * output))).sum(axis = -1)

        # Now, compute the real (squashed) output
        output = torch.tanh(output)
        output = self.output_upper_bounds * output # <-- not sure what this does

        return output, output_log_prob