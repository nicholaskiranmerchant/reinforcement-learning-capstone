'''
The core logic for this rolling buffer comes from
https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py
'''

# Core imports
import numpy as np
from collections import deque

# Typing imports
from typing import Final, Dict

class ReplayBuffer(object):
    def __init__(self,
        state_dim : int,
        action_dim : int,
        capacity : int
        ):
        self._state_dim : Final = state_dim
        self._action_dim : Final = action_dim
        self.capacity = capacity
        self.num_transitions = 0

        self._states = np.zeros((capacity, state_dim))
        self._actions = np.zeros((capacity, action_dim))
        self._next_states = np.zeros((capacity, state_dim))
        self._rewards = np.zeros((capacity,))
        self._terminals = np.zeros((capacity,))

        self._ptr = 0
    
    ### Public Methods ###
    def save_transition(self,
        state : np.ndarray, 
        action : np.ndarray,
        next_state : np.ndarray,
        reward : float,
        terminal : bool
        ) -> None:

        assert state.shape == (self._state_dim,)
        assert action.shape == (self._action_dim,)
        assert next_state.shape == (self._state_dim,)

        self._states[self._ptr,:] = state
        self._actions[self._ptr,:] = action
        self._next_states[self._ptr,:] = next_state
        self._rewards[self._ptr] = reward
        self._terminals[self._ptr] = terminal

        self._ptr = (self._ptr + 1) % self.capacity
        self.num_transitions = np.min([self.num_transitions + 1, self.capacity])

    def sample_transition_batch(self, batch_size : int) -> Dict[str,np.ndarray]:
        states = np.zeros((batch_size, self._state_dim))
        actions = np.zeros((batch_size, self._action_dim))
        next_states = np.zeros((batch_size, self._state_dim))
        rewards = np.zeros((batch_size,))
        terminals = np.zeros((batch_size,))

        for j in range(batch_size):
            i = np.random.randint(0, self.num_transitions)
            
            states[j,:] = self._states[i,:]
            actions[j,:] = self._actions[i,:]
            next_states[j,:] = self._next_states[i,:]
            rewards[j] = self._rewards[i]
            terminals[j] = self._terminals[i]

        return {
            "states": states,
            "actions": actions,
            "next_states": next_states,
            "rewards" : rewards,
            "terminals": terminals
        }
        
    

