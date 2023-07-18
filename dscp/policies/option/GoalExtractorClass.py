# Core imports
import numpy as np

# Typing imports
from abc import ABC, abstractmethod

class GoalExtractor(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_state_dim(self) -> int:
        pass

    @abstractmethod
    def get_goal_dim(self) -> int:
        pass

    @abstractmethod
    def _extract(state : np.ndarray) -> np.ndarray:
        pass

    def extract(state : np.ndarray) -> np.ndarray:
        assert state.shape == (self.get_state_dim(),)
        goal = self._extract(state)
        assert goal.shape == (self.get_goal_dim(),)

        return goal