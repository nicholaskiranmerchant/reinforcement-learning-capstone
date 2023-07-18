# Core imports
import numpy as np

# Typing imports
from abc import ABC, abstractmethod

class MDP(ABC):
    def __init__(self):
        super().__init__()

    ### Public Methods ###
    def act(self, action : np.ndarray) -> None:
        assert action.shape == (self.get_action_dim(),)
        return self._act(action)

    def get_state(self) -> np.ndarray:
        state = self._get_state()
        assert state.shape == (self.get_state_dim(),)
        return state

    def get_state_dim(self) -> int:
        return self.get_state_bounds().shape[0]

    def get_action_dim(self) -> int:
        return self.get_action_bounds().shape[0]

    ### Abstract methods ###
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def _act(self, action : np.ndarray) -> None:
        pass

    @abstractmethod
    def _get_state(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_reward(self) -> float:
        pass

    @abstractmethod
    def get_terminal(self) -> bool:
        pass

    @abstractmethod
    def get_state_bounds(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_action_bounds(self) -> np.ndarray:
        pass