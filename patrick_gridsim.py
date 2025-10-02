# icecream_gridworld.py
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

Action = str  # "Up","Down","Left","Right","Stay"

ACTIONS: List[Action] = ["Up", "Down", "Left", "Right", "Stay"]
DELTA: Dict[Action, Tuple[int, int]] = {
    "Up": (0, 1),
    "Down": (0, -1),
    "Right": (1, 0),
    "Left": (-1, 0),
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
        ax.clear()
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

# ---------- Arrow-key controller ----------

KEY_TO_ACTION: Dict[str, Action] = {
    "up": "Up",
    "down": "Down",
    "left": "Left",
    "right": "Right",
    " ": "Stay",     # space bar
}

class KeyboardController:
    """
    Opens a Matplotlib window. Each arrow key press performs one step.
    'r' resets to the start; 'q' quits.
    """
    def __init__(self, env: IceCreamGridworld):
        self.env = env
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.update(title="Use arrow keys. Space=Stay, r=reset, q=quit")
        plt.show(block=True)

    def update(self, title=""):
        self.env.render(ax=self.ax, title=title)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        key = (event.key or "").lower()
        if key == "q":
            plt.close(self.fig)
            return
        if key == "r":
            self.env.state = self.env.spec.start
            print("\n[reset] state ->", self.env.state)
            self.update(title="Reset")
            return

        if key in KEY_TO_ACTION:
            a = KEY_TO_ACTION[key]
            s_prev = self.env.state
            s_next, o, probs = self.env.step(a)

            # print transition model & outcome
            print(f"\nAction={a} from s={s_prev}")
            for k, v in probs.items():
                print(f"  P(s'={k}) = {v:.3f}")
            print(f"Sampled s'={s_next}, observation o={o}")

            self.update(title=f"a={a}, o={o}")
        else:
            # ignore other keys but keep focus text
            self.update(title="Use arrow keys. Space=Stay, r=reset, q=quit")

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
    KeyboardController(env)  # launches interactive window
