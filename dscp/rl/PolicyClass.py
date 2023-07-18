# Core imports
import numpy as np

# Local imports
from dscp.rl.MDPClass import MDP

# Typing imports
from abc import ABC, abstractmethod

class Policy(ABC):
    def __init__(self,
            state_dim : int,
            action_dim : int,
            ):
        super().__init__()

        self._state_dim = state_dim
        self._action_dim = action_dim

    ### Public Methods ### 
    def sample_action(self, state : np.ndarray) -> np.ndarray:
        assert state.shape == (self._state_dim,)
        action = self._sample_action(state)

        assert action.shape == (self._action_dim,)
        return action
    
    def add_transition_to_buffer(self,
            state: np.ndarray,
            action: np.ndarray,
            next_state: np.ndarray,
            reward: float,
            terminal: bool
            ) -> None:
        assert state.shape == (self._state_dim,)
        assert action.shape == (self._action_dim,)
        assert next_state.shape = (self._state_dim,)

        self._add_transition_to_buffer(
                state,
                action,
                next_state,
                reward,
                terminal
                )

    def get_state_action_probability(
            state: np.ndarray,
            action: np.ndarray,
            ) -> float:
        assert state.shape == (self._state_dim,)
        assert action.shape == (self._action_dim,)

        return self._get_state_action_probability(
                        state, 
                        action
                        )

    def get_state_action_value(
            state: np.ndarray,
            action: np.ndarray,
            ) -> float:
        assert state.shape == (self._state_dim,)
        assert action.shape == (self._action_dim,)

        return self._get_state_action_value(
                        state, 
                        action
                        )

    ### Abstract Methods ###
    @abstractmethod
    def _sample_action(self, state : np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _add_transition_to_buffer(self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        terminal: bool
        ) -> None:
        pass

    @abstractmethod
    def update_from_buffer(self) -> None:
        pass

    @abstractmethod
    def get_state_action_probability(
            state: np.ndarray,
            action: np.ndarray
            ) -> float:
        pass

    @abstractmethod
    def get_state_action_value(
            state: np.ndarray,
            action: np.ndarray
            ) -> float:
        pass