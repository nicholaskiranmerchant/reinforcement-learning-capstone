# Core imports
import numpy as np
import gym

# Local imports
from dscp.rl.MDPClass import MDP

# Typing imports
from typing import Final
from abc import abstractmethod

class GymMDP(MDP):
    def __init__(self, env_name : str):
        self._env : Final = gym.make(env_name)
        self.reset()

        # Assert that this is a continuous environment
        assert isinstance(self._env.observation_space, gym.spaces.Box)
        assert isinstance(self._env.action_space, gym.spaces.Box)

        super().__init__()

    ### Public Methods ###
    def reset(self) -> None:
        self._obs = self._env.reset()
        self._reward : float = 0.
        self._done : bool = False

    def _act(self, action : np.ndarray) -> None:
        obs, reward, done, _ = self._env.step(action)
        self._obs = obs
        self._reward = reward
        self._done = done

    def _get_state(self) -> np.ndarray:
        return self._obs

    def get_reward(self) -> float:
        return self._reward

    def get_terminal(self) -> bool:
        return self._done

    def get_state_bounds(self) -> np.ndarray:
        return self._box_to_ndarray(self._env.observation_space)

    def get_action_bounds(self) -> np.ndarray:
        return self._box_to_ndarray(self._env.action_space)
        
    ### Private Methods ###
    def _box_to_ndarray(self, box : gym.spaces.Box) -> np.ndarray:
        return np.transpose(np.vstack([box.low, box.high]))