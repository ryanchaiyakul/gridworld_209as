import numpy as np
import gridworld

def test_hello():
    s0 = gridworld.Gridworld(10, robot=np.array([0, 0]))
    s1 = gridworld.Gridworld(10, robot=np.array([1, 0]))
    s2 = gridworld.Gridworld(10, robot=np.array([0, 0]))

    assert np.allclose(gridworld.get_transition_probability(gridworld.Action.DOWN, s0, s1), 0.8)
    assert np.allclose(gridworld.get_transition_probability(gridworld.Action.DOWN, s0, s2), 0.15)
    assert np.allclose(gridworld.get_transition_probability(gridworld.Action.STAY, s0, s2), 0.9)

    s0 = gridworld.Gridworld(10, robot=np.array([0, 0]), obstacles=np.array([[0,1]]))
    s1 = gridworld.Gridworld(10, robot=np.array([1, 0]))
    s2 = gridworld.Gridworld(10, robot=np.array([0, 0]))

    assert np.allclose(gridworld.get_transition_probability(gridworld.Action.DOWN, s0, s1), 0.8)
    assert np.allclose(gridworld.get_transition_probability(gridworld.Action.DOWN, s0, s2), 0.2)
    assert np.allclose(gridworld.get_transition_probability(gridworld.Action.STAY, s0, s2), 0.95)
