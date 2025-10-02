# icecream_gridworld.py
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import math
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import patches

Action = str  # "N","S","E","W","Stay"

ACTIONS: List[Action] = ["N", "S", "E", "W", "Stay"]
DELTA: Dict[Action, Tuple[int, int]] = {
    "N": (0, 1),
    "S": (0, -1),
    "E": (1, 0),
    "W": (-1, 0),
    "Stay": (0, 0),
}

@dataclass
class GridSpec:
    width: int
    height: int
    obstacles: set                  # set of (x, y)
    RD: Tuple[int, int]             # ice cream shop D
    RS: Tuple[int, int]             # ice cream shop S
    start: Tuple[int, int]
    p_error: float = 0.2            # transition noise
    seed: Optional[int] = 0

class IceCreamGridworld:
    """
    Canonical gridworld with:
      - actions: N, S, E, W, Stay
      - transition noise: with prob (1 - p_error) the chosen action succeeds,
        with prob p_error one of the 4 alternate outcomes is chosen uniformly.
      - invalid destinations (off grid or obstacle) cause the agent to stay.
      - observation o is a noisy integer based on the harmonic mean h of the
        Euclidean distances to RD and RS, with probabilistic rounding.
    """
    def __init__(self, spec: GridSpec):
        self.spec = spec
        self.rng = np.random.default_rng(spec.seed)
        self.state = spec.start

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.spec.width and 0 <= y < self.spec.height

    def _valid(self, pos: Tuple[int, int]) -> bool:
        return self._in_bounds(*pos) and (pos not in self.spec.obstacles)

    def _apply_delta(self, s: Tuple[int, int], d: Tuple[int, int]) -> Tuple[int, int]:
        cand = (s[0] + d[0], s[1] + d[1])
        return cand if self._valid(cand) else s

    def transition_probs(self, s: Tuple[int, int], a: Action) -> Dict[Tuple[int, int], float]:
        """Return P(s'|s,a) as a dict over successor states."""
        chosen_next = self._apply_delta(s, DELTA[a])
        alts = [x for x in ACTIONS if x != a]
        alt_nexts = [self._apply_delta(s, DELTA[aa]) for aa in alts]

        probs: Dict[Tuple[int, int], float] = {}
        p_succ = 1.0 - self.spec.p_error
        p_err_each = self.spec.p_error / 4.0

        for nxt, p in [(chosen_next, p_succ)] + list(zip(alt_nexts, [p_err_each] * 4)):
            probs[nxt] = probs.get(nxt, 0.0) + p
        return probs

    def step(self, a: Action):
        """
        Sample next state using the model and return:
          next_state, observation_o, probs_dict
        """
        s = self.state
        probs = self.transition_probs(s, a)
        next_states = list(probs.keys())
        pvals = np.array([probs[ns] for ns in next_states], dtype=float)
        pvals /= pvals.sum()
        idx = self.rng.choice(np.arange(len(next_states)), p=pvals)
        s_next = next_states[idx]
        self.state = s_next
        o = self.observe(s_next)
        return s_next, o, probs

    def observe(self, s: Tuple[int, int]) -> int:
        """
        h = 2 / (1/d_D + 1/d_S) with h=0 if either distance is 0.
        Observation o is floor/ceil(h) with probabilistic rounding.
        """
        dD = self._euclid(s, self.spec.RD)
        dS = self._euclid(s, self.spec.RS)
        if dD == 0 or dS == 0:
            h = 0.0
        else:
            h = 2.0 / (1.0 / dD + 1.0 / dS)
        u = math.ceil(h)
        l = math.floor(h)
        if u == l:
            return int(h)
        p_up = h - l
        return u if self.rng.random() < p_up else l

    @staticmethod
    def _euclid(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def render(self, ax=None, title: str = ""):
        close_ax = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))
            close_ax = True

        w, h = self.spec.width, self.spec.height
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect("equal")
        ax.set_xticks(range(w + 1))
        ax.set_yticks(range(h + 1))
        ax.grid(True)

        # obstacles
        for (ox, oy) in self.spec.obstacles:
            ax.add_patch(patches.Rectangle((ox, oy), 1, 1, hatch="x", fill=False))

        # shops
        for lbl, pos in [("RD", self.spec.RD), ("RS", self.spec.RS)]:
            ax.add_patch(patches.Rectangle((pos[0], pos[1]), 1, 1, fill=False))
            ax.text(pos[0] + 0.5, pos[1] + 0.5, lbl, ha="center", va="center")

        # agent
        ax.add_patch(patches.Circle((self.state[0] + 0.5, self.state[1] + 0.5), 0.28))

        if title:
            ax.set_title(title)
        if close_ax:
            plt.show()

def run_demo(env: IceCreamGridworld, actions: List[Action]):
    env.state = env.spec.start
    for t, a in enumerate(actions):
        s_prev = env.state
        s_next, o, probs = env.step(a)
        print(f"Step {t}: s={s_prev}, a={a}")
        for k, v in probs.items():
            print(f"  -> {k}: {v:.3f}")
        print(f"  Sampled s'={s_next}, observation o={o}\n")
        fig, ax = plt.subplots(figsize=(4, 4))
        env.render(ax=ax, title=f"t={t}, action={a}, o={o}")
        plt.show()
    fig, ax = plt.subplots(figsize=(4, 4))
    env.render(ax=ax, title=f"Final state: {env.state}")
    plt.show()

if __name__ == "__main__":
    # Example map matching the handout figure
    spec = GridSpec(
        width=5, height=5,
        obstacles={(1, 1), (1, 3), (3, 1), (3, 3)},
        RD=(2, 2),
        RS=(1, 0),
        start=(2, 4),
        p_error=0.2,
        seed=42,
    )
    env = IceCreamGridworld(spec)

    # Try your own actions here
    actions = ["W", "W", "S", "S", "E", "E", "N", "N", "Stay"]
    run_demo(env, actions)
