# Core imports
import numpy as np
from copy import deepcopy
from itertools import chain
import torch

# Local imports
from dscp.rl.PolicyClass import Policy
from dscp.rl.MDPClass import MDP
from dscp.rl.ReplayBufferClass import ReplayBuffer
from dscp.policies.sac.QNetworkClass import QNetwork
from dscp.policies.sac.SquashedGaussianNetworkClass import SquashedGaussianNetwork

# Typing imports
from typing import List, Dict

'''
Some reccomended hyperparameter values:
gamma - 0.9
replay_buffer_cap - 300,000
update_batch_size - 128
q_learning_rate - 0.001
pi_learning_rate - 0.001
loss_alpha - 0.2???
polyak_rate - 0.995
'''

class SAC(Policy):
    def __init__(self,
            state_dim: int,
            action_dim: int,
            action_bounds: np.ndarray,

            # SAC hyperparameters
            gamma: float = 0.99,
            update_batch_size: int = 128,
            loss_alpha: float = 0.2, # This might be wrong
            polyak_rate: float = 0.995,

            # Replay Buffer hyperparameters
            replay_buffer_cap: int = 300000,

            # Q network hyperparameters
            q_hidden_1_dim: int = 256,
            q_hidden_2_dim: int = 256,
            q_learning_rate: float = 0.001,

            # Pi network hyperparameters
            pi_learning_rate: float
            ):
        super().__init__(state_dim, action_dim)

        assert action_bounds.shape == (action_dim, 2)
        self.action_bounds = action_bounds

        self._gamma = gamma

        assert replay_buffer_cap >= 0
        self.replay_buffer = ReplayBuffer(
            state_dim,
            action_dim,
            replay_buffer_cap
            )   

        assert 0 < update_batch_size <= replay_buffer_cap
        self._update_batch_size = update_batch_size

        self._loss_alpha = loss_alpha
        self._polyak_rate = polyak_rate

        # Initialize source networks
        self.q1 = QNetwork(
                    state_dim, 
                    action_dim,
                    q_hidden_1_dim,
                    q_hidden_2_dim
                    )

        self.q2 = QNetwork(
                    state_dim, 
                    action_dim,
                    q_hidden_1_dim,
                    q_hidden_2_dim
                    )

        self.pi = SquashedGaussianNetwork(
                    state_dim, 
                    action_dim,
                    action_bounds[:,1]
                    )

        # Initialize tracking target networks
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)
        self.target_pi = deepcopy(self.pi)

        self.q_params = chain(
                self.q1.parameters(), 
                self.q2.parameters()
                )
        self.pi_params = self.pi.parameters()
        self.params = chain(self.q_params, self.pi_params)

        self.target_params = chain(
                self.target_q1.parameters(),
                self.target_q2.parameters(),
                self.target_pi.parameters()
                )

        # Initializer optimizers
        self.q_optimizer = torch.optim.Adam(self.q_params, lr=q_learning_rate)
        self.pi_optimizer = torch.optim.Adam(self.pi_params, lr=pi_learning_rate)

    

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, batch : Dict[str,torch.Tensor]) -> torch.Tensor:

        q1_out = self.q1(batch["states"], batch["actions"])
        q2_out = self.q2(batch["states"], batch["actions"])

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            # a2, logp_a2 = ac.pi(o2)
            next_actions, next_action_log_probs = self.pi(batch["next_states"])

            # Target Q-values
            q1_pi_target = self.target_q1(batch["next_states"], next_actions)
            q2_pi_target = self.target_q2(batch["next_states"], next_actions)
            q_pi_target = torch.min(q1_pi_target, q2_pi_target)

            backup = batch["rewards"] + self._gamma * (1 - batch["terminals"]) * (q_pi_target - self._loss_alpha * next_action_log_probs)

        # MSE loss against Bellman backup
        loss_q1 = ((q1_out - backup)**2).mean()
        loss_q2 = ((q2_out - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    def compute_loss_pi(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        actions, action_log_probs = self.pi(batch["states"])
        q1_pi = self.q1(batch["states"], actions)
        q2_pi = self.q2(batch["states"], actions)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss

        loss_pi = (self._loss_alpha * action_log_probs - q_pi).mean()

        return loss_pi

    def _update(self, batch: Dict[str, torch.Tensor]):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(batch)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(batch)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.params, self.target_params):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self._polyak_rate)
                p_targ.data.add_((1 - self._polyak_rate) * p.data)

    ### INHERITED METHODS ###
    def _sample_action(self, state : np.ndarray) -> np.ndarray:
        state = torch.tensor(state, dtype=torch.float32)
        action, _ = self.pi(state)
        action = action.detach().numpy()

        return action

    def add_transition_to_buffer(self,
            state: np.ndarray,
            action: np.ndarray,
            next_state: np.ndarray,
            reward: float,
            terminal: bool
            ) -> None:

        self.replay_buffer.save_transition(
            state, 
            action, 
            next_state, 
            reward, 
            terminal
            )

    def update_from_buffer(self) -> None:
        if self._update_batch_size <= self.replay_buffer.num_transitions:
            batch = self.replay_buffer.sample_transition_batch(self._update_batch_size)
            for key in batch.keys():
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)

            self._update(batch)



