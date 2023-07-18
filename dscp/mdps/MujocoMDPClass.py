# Core imports
import numpy as np
import mujoco_py

# Local imports
from dscp.rl.MDPClass import MDP

# Typing imports
from abc import abstractmethod
from typing import Final

class MujocoMDP(MDP):
    def __init__(self, model_path : str, frame_skip : int):
        self._model = mujoco_py.load_model_from_path(model_path)
        self._sim = mujoco_py.MjSim(self._model)
        self._frame_skip : Final = frame_skip

        self._init_qpos : Final[np.ndarray] = self._sim.data.qpos.ravel().copy()
        self._init_qvel : Final[np.ndarray] = self._sim.data.qvel.ravel().copy()

        super().__init__()

    #### Methods ####

    def reset(self) -> None:
        # Sample near initial states
        qpos = self._init_qpos + np.random.uniform(size=self._model.nq, low=-.1, high=.1)
        qvel = self._init_qvel + np.random.randn(self._model.nv) * .1 

        # Forward the new state to themodel
        old_state = self._sim.get_state()
        new_state = mujoco_py.MjSimState(
            old_state.time, 
            qpos,
            qvel,
            old_state.act, 
            old_state.udd_state
            )
        self._sim.set_state(new_state)
        self._sim.forward()

    def _act(self, action : np.ndarray) -> None:
        assert self._sim.data.ctrl.shape == action.shape
        self._sim.data.ctrl[:] = action
        for _ in range(self._frame_skip):
            self._sim.step()

    def _get_state(self) -> np.ndarray:
        return np.concatenate([
            self._sim.data.qpos.flat,
            self._sim.data.qvel.flat
        ])


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