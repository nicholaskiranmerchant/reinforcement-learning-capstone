import ipdb
import numpy as np
import torch
import torch.nn.functional as F

from dscp.policies.td3.replay_buffer import ReplayBuffer
from dscp.policies.td3.model import Actor, Critic
from dscp.policies.td3.utils import *

from dscp.rl.PolicyClass import Policy


# Adapted author implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class TD3(Policy):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            batch_size=256,
            exploration_noise=0.1,
            exploration_method="",
            device=torch.device("cuda")
    ):
        super().__init__(state_dim, action_dim)

        assert exploration_method in ("", "shaping"), exploration_method

        self.actor = Actor(state_dim, action_dim, max_action)#.to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.replay_buffer = ReplayBuffer(state_dim, action_dim)

        self.max_action = max_action
        self.action_dim = action_dim
        self.gamma = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.batch_size = batch_size
        self.epsilon = exploration_noise
        self.exploration_method = exploration_method
        self.device = device

        self.trained_options = []
        self.critic_learning_rate = 3e-4
        self.actor_learning_rate = 3e-4

        self.total_it = 0

    def act(self, state, evaluation_mode=False):
        state = torch.FloatTensor(state.reshape(1, -1))#.to(self.device)
        selected_action = self.actor(state).cpu().data.numpy().flatten()
        noise = np.random.normal(0, self.max_action * self.epsilon, size=self.action_dim)
        if not evaluation_mode:
            selected_action += noise
        return selected_action.clip(-self.max_action, self.max_action)

    def step(self, state, action, reward, next_state, is_terminal):
        self.replay_buffer.add(state, action, reward, next_state, is_terminal)

        if len(self.replay_buffer) > self.batch_size:
            self.train(self.replay_buffer, self.batch_size)

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer - result is tensors
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        if len(self.trained_options) > 0 and self.exploration_method == "shaping":
            shaping_bonus = self.get_exploration_bonus(next_state)
            assert shaping_bonus.shape == reward.shape
            reward += shaping_bonus

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1. - done) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_epsilon(self):
        """ We are using fixed (default) epsilons for TD3 because tuning it is hard. """
        pass

    def get_qvalues(self, states, actions):
        self.critic.eval()
        with torch.no_grad():
            q_values = self.critic.Q1(states, actions)
        self.critic.train()
        return q_values

    def get_exploration_bonus(self, next_states):
        """
        Optional exploration bonus for reaching certain regions of the state-space.

        Args:
            next_states (torch.tensor)

        Returns:
            bonuses (torch.tensor)
        """
        np_next_states = next_states.clone().detach().cpu().numpy()
        shaping_rewards = np.zeros((np_next_states.shape[0],))
        for option in self.trained_options:
            if option.should_target_with_bonus():
                shaping_rewards += option.batched_is_init_true(np_next_states)
        return torch.FloatTensor(shaping_rewards).unsqueeze(1)#.to(self.device).unsqueeze(1)

    # Kiran's new compat function
    def _sample_action(self, state : np.ndarray) -> np.ndarray:
        return self.act(state)

    def add_transition_to_buffer(self,
            state: np.ndarray,
            action: np.ndarray,
            next_state: np.ndarray,
            reward: float,
            terminal: bool
            ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, terminal)

    def update_from_buffer(self) -> None:
        if len(self.replay_buffer) > self.batch_size:
            self.train(self.replay_buffer, self.batch_size)


