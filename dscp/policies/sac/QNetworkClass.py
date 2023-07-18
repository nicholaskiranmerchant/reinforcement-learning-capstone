# Core imports
import numpy as np
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self,
            state_dim: int,
            action_dim: int,
            hidden_1_dim: int,
            hidden_2_dim: int,
            ):
        super().__init__()

        self.q = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_1_dim),
            nn.ReLU(),
            nn.Linear(hidden_1_dim, self.hidden_2_dim),
            nn.ReLU(),
            nn.Linear(hidden_2_dim, 1),
            )

    ### Public Methods ###
    def forward(self, 
            state: torch.Tensor, 
            action: torch.Tensor
            ) -> torch.Tensor:
            
        q_value = self.q(torch.cat([state, action], dim=-1))
        q_value = torch.squeeze(q_value, -1)
        return q_value
