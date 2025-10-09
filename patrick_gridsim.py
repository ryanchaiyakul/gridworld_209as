# icecream_gridworld_mdp.py  (with red road cells + adjustable penalty)
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

# --------------------------- Core env ---------------------------

Action = str  # "Up","Down","Left","Right","Stay"
ACTIONS: List[Action] = ["Up", "Down", "Left", "Right", "Stay"]
DELTA: Dict[Action, Tuple[int, int]] = {
    "Up": (0, 1),
    "Down": (0, -1),
    "Right": (1, 0),
    "Left": (-1, 0),
    "Stay": (0, 0),
}

# ---- tie-break helpers (to prefer shortest path at iteration 1) ----
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def nearest_store_distance(env, s):
    return min(manhattan(s, env.spec.RD), manhattan(s, env.spec.RS))

def tie_break_by_distance(env, s, a, use_expected=True):
    """
    Lower is better. If use_expected=True, use expected Manhattan distance
    after the stochastic step. If False, use only the deterministic outcome.
    """
    if use_expected:
        probs = env.transition_probs(s, a)
        return sum(p * nearest_store_distance(env, s_next) for s_next, p in probs.items())
    else:
        det = env._apply_delta(s, DELTA[a])
        return nearest_store_distance(env, det)

@dataclass
class GridSpec:
    width: int
    height: int
    obstacles: set                  # blocked cells (impassable)
    RD: Tuple[int, int]             # shop D
    RS: Tuple[int, int]             # shop S
    start: Tuple[int, int]
    p_error: float = 0.2            # transition noise
    seed: Optional[int] = 0
    roads: set = None               # NEW: passable road cells with penalty

    def __post_init__(self):
        if self.roads is None:
            self.roads = set()

class IceCreamGridworld:
    """
    5-action gridworld with transition noise.
    Obstacles block; roads are passable but penalized (handled by reward).
    Observation = probabilistic rounding of harmonic-mean distance to RD and RS.
    """
    def __init__(self, spec: GridSpec):
        self.spec = spec
        self.rng = np.random.default_rng(spec.seed)
        self.state = spec.start

    # --- helpers
    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.spec.width and 0 <= y < self.spec.height

    def _valid(self, pos: Tuple[int, int]) -> bool:
        return self._in_bounds(*pos) and (pos not in self.spec.obstacles)

    def _apply_delta(self, s: Tuple[int, int], d: Tuple[int, int]) -> Tuple[int, int]:
        cand = (s[0] + d[0], s[1] + d[1])
        return cand if self._valid(cand) else s

    # --- transition
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
        """Sample one step and return (s_next, observation, probs dict)."""
        s = self.state
        probs = self.transition_probs(s, a)
        next_states = list(probs.keys())
        pvals = np.array([probs[ns] for ns in next_states], dtype=float)
        pvals /= pvals.sum()
        s_next = next_states[self.rng.choice(np.arange(len(next_states)), p=pvals)]
        self.state = s_next
        o = self.observe(s_next)
        return s_next, o, probs

    # --- observation model
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
        return u if self.rng.random() < (h - l) else l

    @staticmethod
    def _euclid(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # --- rendering
    def render(self, ax=None, title: str = ""):
        close_ax = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(4.8, 4.8))
            close_ax = True

        w, h = self.spec.width, self.spec.height
        ax.clear()
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect("equal")
        ax.set_xticks(range(w + 1))
        ax.set_yticks(range(h + 1))
        ax.grid(True)

        # Draw roads (red fill, passable)
        for (rx, ry) in self.spec.roads:
            ax.add_patch(patches.Rectangle((rx, ry), 1, 1, facecolor="red", alpha=0.25, edgecolor="none"))

        # Obstacles (hatched boxes)
        for (ox, oy) in self.spec.obstacles:
            ax.add_patch(patches.Rectangle((ox, oy), 1, 1, hatch="x", fill=False))

        # Shops
        for lbl, pos in [("RD", self.spec.RD), ("RS", self.spec.RS)]:
            ax.add_patch(patches.Rectangle((pos[0], pos[1]), 1, 1, fill=False))
            ax.text(pos[0] + 0.5, pos[1] + 0.5, lbl, ha="center", va="center")

        # Agent
        ax.add_patch(patches.Circle((self.state[0] + 0.5, self.state[1] + 0.5), 0.28))
        if title:
            ax.set_title(title)
        if close_ax:
            plt.show()

# --------------------------- Keyboard control ---------------------------

KEY_TO_ACTION: Dict[str, Action] = {
    "up": "Up",
    "down": "Down",
    "left": "Left",
    "right": "Right",
    " ": "Stay",   # space bar
}

class KeyboardController:
    """
    Matplotlib window: each key press performs one step.
    Space = Stay, r = reset, q = quit.
    """
    def __init__(self, env: IceCreamGridworld):
        self.env = env
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.update("Arrow keys to move. Space=Stay, r=reset, q=quit")
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
            print("\n[reset] state:", self.env.state)
            self.update("Reset")
            return
        if key in KEY_TO_ACTION:
            a = KEY_TO_ACTION[key]
            s_prev = self.env.state
            s_next, o, probs = self.env.step(a)
            print(f"\nAction={a} from s={s_prev}")
            for k, v in probs.items():
                print(f"  P(s'={k}) = {v:.3f}")
            print(f"Sampled s'={s_next}, observation o={o}")
            self.update(f"a={a}, o={o}")
        else:
            self.update("Arrow keys to move. Space=Stay, r=reset, q=quit")

# --------------------------- MDP planner ---------------------------

@dataclass
class TaskSpec:
    RD: float = +10.0
    RS: float = +10.0
    RW: float = -0.1          # NEW: penalty when current state is a road cell
    gamma: float = 0.95
    reward_mode: str = "start_state"  # "start_state" | "intentional_entry" | "terminal_once"

State = Tuple[int, int]

def enumerate_states(env) -> List[State]:
    return [(x, y)
            for x in range(env.spec.width)
            for y in range(env.spec.height)
            if (x, y) not in env.spec.obstacles]

def is_store(env, s: State) -> Optional[str]:
    if s == env.spec.RD:
        return "RD"
    if s == env.spec.RS:
        return "RS"
    return None

# ---------- Reward models (roads penalize based on *current* state) ----------
def on_road(env, s: State) -> bool:
    return s in env.spec.roads

def reward_start_state(env, task: TaskSpec, s: State) -> float:
    t = is_store(env, s)
    if t == "RD":
        return task.RD
    if t == "RS":
        return task.RS
    return task.RW if on_road(env, s) else 0.0

def reward_intentional_entry(env, task: TaskSpec, s: State, a: Action, s_next: State) -> float:
    # Step cost from being on a road at the *start* of the step
    step_cost = task.RW if on_road(env, s) else 0.0
    det = env._apply_delta(s, DELTA[a])
    if s_next == det:
        t = is_store(env, s_next)
        if t == "RD":
            return step_cost + task.RD
        if t == "RS":
            return step_cost + task.RS
    return step_cost

def reward_terminal_once(env, task: TaskSpec, s: State, s_next: State) -> Tuple[float, bool]:
    # If entering a store, pay reward once and terminate; otherwise step cost if starting on road.
    t = is_store(env, s_next)
    if t == "RD":
        return (task.RD, True)
    if t == "RS":
        return (task.RS, True)
    return ((task.RW if on_road(env, s) else 0.0), False)

# ---------- Value iteration (with shortest-path tie-break) ----------
def value_iteration(env, task: TaskSpec, theta: float = 1e-6, max_iter: int = 10000):
    S = enumerate_states(env)
    idx = {s: i for i, s in enumerate(S)}
    V = np.zeros(len(S), dtype=float)
    terminal = (task.reward_mode == "terminal_once")
    eps = 1e-12

    for _ in range(max_iter):
        delta = 0.0
        V_new = V.copy()
        for s in S:
            i = idx[s]
            if terminal and is_store(env, s):
                V_new[i] = 0.0
                continue
            best_q, best_tie = -1e18, +1e18
            for a in ACTIONS:
                probs = env.transition_probs(s, a)
                q = 0.0
                for s_next, p in probs.items():
                    if task.reward_mode == "start_state":
                        r, done = reward_start_state(env, task, s), False
                    elif task.reward_mode == "intentional_entry":
                        r, done = reward_intentional_entry(env, task, s, a, s_next), False
                    else:
                        r, done = reward_terminal_once(env, task, s, s_next)
                    j = idx[s_next]
                    nxt = 0.0 if (terminal and done) else V[j]
                    q += p * (r + task.gamma * nxt)
                tie = tie_break_by_distance(env, s, a, use_expected=True)
                if (q > best_q + eps) or (abs(q - best_q) <= eps and tie < best_tie):
                    best_q, best_tie = q, tie
            V_new[i] = best_q
            delta = max(delta, abs(V_new[i] - V[i]))
        V = V_new
        if delta < theta:
            break

    # Greedy policy (same tie-break)
    pi: Dict[State, Action] = {}
    for s in S:
        if terminal and is_store(env, s):
            pi[s] = "Stay"
            continue
        best_a, best_q, best_tie = "Stay", -1e18, +1e18
        for a in ACTIONS:
            probs = env.transition_probs(s, a)
            q = 0.0
            for s_next, p in probs.items():
                if task.reward_mode == "start_state":
                    r, done = reward_start_state(env, task, s), False
                elif task.reward_mode == "intentional_entry":
                    r, done = reward_intentional_entry(env, task, s, a, s_next), False
                else:
                    r, done = reward_terminal_once(env, task, s, s_next)
                j = idx[s_next]
                nxt = 0.0 if (terminal and done) else V[j]
                q += p * (r + task.gamma * nxt)
            tie = tie_break_by_distance(env, s, a, use_expected=True)
            if (q > best_q + eps) or (abs(q - best_q) <= eps and tie < best_tie):
                best_q, best_a, best_tie = q, a, tie
        pi[s] = best_a
    return V, pi, S

# ---------- Policy iteration (with tie-break) ----------
def policy_iteration(env, task: TaskSpec, theta: float = 1e-8, max_eval: int = 10000, max_iter: int = 1000):
    S = enumerate_states(env)
    idx = {s: i for i, s in enumerate(S)}
    pi = {s: "Stay" for s in S}
    V = np.zeros(len(S), dtype=float)
    terminal = (task.reward_mode == "terminal_once")
    eps = 1e-12

    def policy_eval():
        for _ in range(max_eval):
            delta = 0.0
            for s in S:
                i = idx[s]
                a = pi[s]
                if terminal and is_store(env, s):
                    newV = 0.0
                else:
                    probs = env.transition_probs(s, a)
                    val = 0.0
                    for s_next, p in probs.items():
                        if task.reward_mode == "start_state":
                            r, done = reward_start_state(env, task, s), False
                        elif task.reward_mode == "intentional_entry":
                            r, done = reward_intentional_entry(env, task, s, a, s_next), False
                        else:
                            r, done = reward_terminal_once(env, task, s, s_next)
                        j = idx[s_next]
                        nxt = 0.0 if (terminal and done) else V[j]
                        val += p * (r + task.gamma * nxt)
                    newV = val
                delta = max(delta, abs(newV - V[i]))
                V[i] = newV
            if delta < theta:
                break

    for _ in range(max_iter):
        policy_eval()
        stable = True
        for s in S:
            old = pi[s]
            best_a, best_q, best_tie = old, -1e18, +1e18
            for a in ACTIONS:
                probs = env.transition_probs(s, a)
                q = 0.0
                for s_next, p in probs.items():
                    if task.reward_mode == "start_state":
                        r, done = reward_start_state(env, task, s), False
                    elif task.reward_mode == "intentional_entry":
                        r, done = reward_intentional_entry(env, task, s, a, s_next), False
                    else:
                        r, done = reward_terminal_once(env, task, s, s_next)
                    j = idx[s_next]
                    nxt = 0.0 if (terminal and done) else V[j]
                    q += p * (r + task.gamma * nxt)
                tie = tie_break_by_distance(env, s, a, use_expected=True)
                if (q > best_q + eps) or (abs(q - best_q) <= eps and tie < best_tie):
                    best_q, best_a, best_tie = q, a, tie
            pi[s] = best_a
            if best_a != old:
                stable = False
        if stable:
            break
    return V, pi, S

# Utilities
ARROW = {"Up": "↑", "Down": "↓", "Left": "←", "Right": "→", "Stay": "•"}

def plot_policy(env, pi, S, title="Policy"):
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    env.render(ax=ax, title=title)
    for s, a in pi.items():
        x, y = s
        ax.text(x + 0.5, y + 0.5, ARROW[a], ha="center", va="center", fontsize=16)
    plt.show()

def rollout(env, pi, start: Optional[State] = None, steps: int = 30, seed: Optional[int] = None):
    if start is None:
        start = env.spec.start
    if seed is not None:
        env.rng = np.random.default_rng(seed)
    env.state = start
    traj = [(env.state, None, None)]
    for _ in range(steps):
        a = pi[env.state]
        s_next, o, _ = env.step(a)
        traj.append((s_next, a, o))
    return traj

# --------------------------- Run as script ---------------------------

if __name__ == "__main__":
    # <<< Adjust your map here >>>
    spec = GridSpec(
        width=5, height=5,
        obstacles={(1, 1), (1, 3), (3, 1), (3, 3)},
        RD=(4, 4),
        RS=(0, 0),
        start=(1, 2),
        p_error=0.2,
        seed=42,
        roads={(0,2), (1,2), (2,2)}  # NEW: red road cells (passable, penalized)
    )
    env = IceCreamGridworld(spec)

    # 1) Interactive driving (uncomment to use)
    # KeyboardController(env)

    # 2) Planning — start-state rewards
    task = TaskSpec(RD=10.0, RS=10.0, RW=-0.2, gamma=0.95, reward_mode="start_state")  # adjust RW here
    V, pi, S = value_iteration(env, task)
    plot_policy(env, pi, S, title="Optimal policy (roads penalized)")

    # Try other modes if you like:
    # task2 = TaskSpec(RD=10.0, RS=10.0, RW=-0.2, gamma=0.95, reward_mode="intentional_entry")
    # V2, pi2, S2 = value_iteration(env, task2)
    # plot_policy(env, pi2, S2, title="Policy (intentional entry + road penalty)")

    # task3 = TaskSpec(RD=20.0, RS=20.0, RW=-0.2, gamma=0.99, reward_mode="terminal_once")
    # V3, pi3, S3 = value_iteration(env, task3)
    # plot_policy(env, pi3, S3, title="Policy (terminal once + road penalty)")
