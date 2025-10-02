from __future__ import annotations
import enum
import random
import numpy as np


class Action(enum.IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4

    def get_alternate(self) -> list[Action]:
        if self == Action.UP:
            return [Action.DOWN, Action.LEFT, Action.RIGHT, Action.STAY]
        if self == Action.DOWN:
            return [Action.UP, Action.LEFT, Action.RIGHT, Action.STAY]
        if self == Action.LEFT:
            return [Action.DOWN, Action.UP, Action.RIGHT, Action.STAY]
        if self == Action.RIGHT:
            return [Action.DOWN, Action.UP, Action.LEFT, Action.STAY]
        if self == Action.STAY:
            return [Action.DOWN, Action.UP, Action.RIGHT, Action.LEFT]
        return []


class Square(enum.IntEnum):
    EMPTY = 0
    ROBOT = 1
    OBSTACLE = 2
    ROAD = 3
    STORE = 4


class Gridworld:
    def __init__(
        self,
        N: int,
        robot: np.ndarray = np.empty((0, 2)),
        obstacles: np.ndarray = np.empty((0, 2)),
        roads: np.ndarray = np.empty((0, 2)),
        stores: np.ndarray = np.empty((0, 2)),
    ) -> None:
        self.N = N
        self.map = np.zeros((N, N))
        self.add_entity(robot[None, ...], Square.ROBOT)
        self.add_entity(obstacles, Square.OBSTACLE)
        self.add_entity(roads, Square.ROAD)
        self.add_entity(stores, Square.STORE)

        self.robot = robot

    def add_entity(self, locations: np.ndarray, value: Square) -> None:
        self.map[self._to_bool_map(locations)] = value

    def _to_bool_map(self, locations: np.ndarray) -> np.ndarray:
        int_locations = locations.astype(int)
        ret = np.zeros_like(self.map, dtype=bool)
        ret[int_locations[:, 0], int_locations[:, 1]] = True
        return ret

    def is_adjacent_to_robot(self, squares: np.ndarray) -> np.ndarray:
        is_adjacent = (
            np.sum(np.abs(squares - self.robot[None, ...]), axis=1) <= 1
        )
        is_valid = (
            (squares[:, 0] >= 0)
            & (squares[:, 0] < self.N)
            & (squares[:, 1] >= 0)
            & (squares[:, 1] < self.N)
        )
        return is_adjacent & is_valid
    
    def get_motion_result(self, action: Action) -> np.ndarray:
        if action == action.UP:
            new_robot = self.robot + np.array([0, -1])
        elif action == action.DOWN:
            new_robot = self.robot + np.array([0, 1])
        elif action == action.RIGHT:
            new_robot = self.robot + np.array([1, 0])
        elif action == action.LEFT:
            new_robot = self.robot + np.array([-1, 0])
        elif action == action.STAY:
            new_robot = self.robot + np.array([0, 0])
        else:
            new_robot = self.robot.copy()
        return np.clip(new_robot, 0, self.N - 1)

def get_transition_probability(
    a: Action, s: Gridworld, s_prime: Gridworld, p_e: float = 0.2
) -> float:
    if not s.is_adjacent_to_robot(s_prime.robot[None, ...]):
        # If the next robot state is not possible
        return 0.0
    if np.all(s_prime.robot == s.robot):
        # Staying in the same place has 2 cases
        # 1. Choose action = STAY, and executed motion (1 - p_e)
        # 2. Choose action = STAY, executed alternate motion, and stayed because of obstacle (p_e * (N/4))
        # 3. Choose alternate motion, and ended up with STAY (p_e)
        # 4. Choose alternate motion, executed alternate motion, and stayed because of obstacle (p_e * (N/4))
        pass
    if np.all(s_prime.robot == s.get_motion_result(a)):
        return 1 - p_e
    return p_e

s0 = Gridworld(10, robot=np.array([0, 0]))
s1 = Gridworld(10, robot=np.array([0, 1]))
s2 = Gridworld(10, robot=np.array([3, 0]))

print(get_transition_probability(Action.DOWN, s0, s1))
#print(get_transition_probability(Action.STAY, s0, s2))