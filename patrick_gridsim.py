# icecream_gridworld_mdp_optimized.py
# Week 1–2 features + Week 3 multi-agent with fast n=2 exact VI
# Ready to run as a single script.

from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Iterable, Callable
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

# --------------------------- Core env (single agent) ---------------------------

Action = str  # "Up","Down","Left","Right","Stay"
ACTIONS: List[Action] = ["Up", "Down", "Left", "Right", "Stay"]
DELTA: Dict[Action, Tuple[int, int]] = {
    "Up": (0, 1),
    "Down": (0, -1),
    "Right": (1, 0),
    "Left": (-1, 0),
    "Stay": (0, 0),
}
A2I = {a: i for i, a in enumerate(ACTIONS)}
I2A = {i: a for a, i in A2I.items()}

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def nearest_store_distance(env, s):
    return min(manhattan(s, env.spec.RD), manhattan(s, env.spec.RS))

@dataclass
class GridSpec:
    width: int
    height: int
    obstacles: set
    RD: Tuple[int, int]
    RS: Tuple[int, int]
    start: Tuple[int, int]
    p_error: float = 0.2
    seed: Optional[int] = 0
    roads: set = None

    def __post_init__(self):
        if self.roads is None:
            self.roads = set()

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
        p_succ = 1.0 - self.spec.p_error
        p_err_each = self.spec.p_error / 4.0

        probs: Dict[Tuple[int, int], float] = {}
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
        u = math.ceil(h); l = math.floor(h)
        if u == l:
            return int(h)
        return u if self.rng.random() < (h - l) else l

    @staticmethod
    def _euclid(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def render(self, ax=None, title: str = ""):
        close_ax = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(4.8, 4.8))
            close_ax = True

        w, h = self.spec.width, self.spec.height
        ax.clear()
        ax.set_xlim(0, w); ax.set_ylim(0, h)
        ax.set_aspect("equal")
        ax.set_xticks(range(w + 1)); ax.set_yticks(range(h + 1))
        ax.grid(True)

        for (rx, ry) in self.spec.roads:
            ax.add_patch(patches.Rectangle((rx, ry), 1, 1, facecolor="red", alpha=0.25, edgecolor="none"))
        for (ox, oy) in self.spec.obstacles:
            ax.add_patch(patches.Rectangle((ox, oy), 1, 1, hatch="x", fill=False))
        for lbl, pos in [("RD", self.spec.RD), ("RS", self.spec.RS)]:
            ax.add_patch(patches.Rectangle((pos[0], pos[1]), 1, 1, fill=False))
            ax.text(pos[0] + 0.5, pos[1] + 0.5, lbl, ha="center", va="center")
        ax.add_patch(patches.Circle((self.state[0] + 0.5, self.state[1] + 0.5), 0.28))
        if title:
            ax.set_title(title)
        if close_ax:
            plt.show()

# --------------------------- Keyboard control ---------------------------

KEY_TO_ACTION: Dict[str, Action] = {
    "up": "Up", "down": "Down", "left": "Left", "right": "Right", " ": "Stay",
}

class KeyboardController:
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
            plt.close(self.fig); return
        if key == "r":
            self.env.state = self.env.spec.start
            print("\n[reset] state:", self.env.state)
            self.update("Reset"); return
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

# --------------------------- MDP planner (single agent) ---------------------------

@dataclass
class TaskSpec:
    RD: float = +10.0
    RS: float = +10.0
    RW: float = -0.1
    gamma: float = 0.95
    reward_mode: str = "start_state"  # "start_state" | "intentional_entry" | "terminal_once"

State = Tuple[int, int]

def enumerate_states(env) -> List[State]:
    return [(x, y)
            for x in range(env.spec.width)
            for y in range(env.spec.height)
            if (x, y) not in env.spec.obstacles]

def is_store(env, s: State) -> Optional[str]:
    if s == env.spec.RD: return "RD"
    if s == env.spec.RS: return "RS"
    return None

def on_road(env, s: State) -> bool:
    return s in env.spec.roads

def reward_start_state(env, task: TaskSpec, s: State) -> float:
    t = is_store(env, s)
    if t == "RD": return task.RD
    if t == "RS": return task.RS
    return task.RW if on_road(env, s) else 0.0

def reward_intentional_entry(env, task: TaskSpec, s: State, a: Action, s_next: State) -> float:
    step_cost = task.RW if on_road(env, s) else 0.0
    det = env._apply_delta(s, DELTA[a])
    if s_next == det:
        t = is_store(env, s_next)
        if t == "RD": return step_cost + task.RD
        if t == "RS": return step_cost + task.RS
    return step_cost

def reward_terminal_once(env, task: TaskSpec, s: State, s_next: State) -> Tuple[float, bool]:
    t = is_store(env, s_next)
    if t == "RD": return (task.RD, True)
    if t == "RS": return (task.RS, True)
    return ((task.RW if on_road(env, s) else 0.0), False)

def value_iteration(env, task: TaskSpec, theta: float = 1e-6, max_iter: int = 10000):
    S = enumerate_states(env); nS = len(S)
    idx = {s: i for i, s in enumerate(S)}
    V = np.zeros(nS, dtype=float)
    terminal = (task.reward_mode == "terminal_once")
    eps = 1e-12

    # Precompute per-state action transitions and tie-breaks
    next_idx = [[None]*len(ACTIONS) for _ in range(nS)]
    next_prob = [[None]*len(ACTIONS) for _ in range(nS)]
    tie_val = np.zeros((nS, len(ACTIONS)), dtype=float)
    for i, s in enumerate(S):
        for a_i, a in enumerate(ACTIONS):
            probs = env.transition_probs(s, a)
            ns = np.array([idx[ss] for ss in probs.keys()], dtype=int)
            ps = np.array(list(probs.values()), dtype=float)
            next_idx[i][a_i] = ns
            next_prob[i][a_i] = ps
            # expected nearest-store distance
            d = np.array([nearest_store_distance(env, ss) for ss in probs.keys()], dtype=float)
            tie_val[i, a_i] = float((ps * d).sum())

    # Gauss–Seidel updates
    for _ in range(max_iter):
        delta = 0.0
        for i, s in enumerate(S):
            if terminal and is_store(env, s):
                newV = 0.0
            else:
                best_q, best_t = -1e18, +1e18
                for a_i, a in enumerate(ACTIONS):
                    ns = next_idx[i][a_i]; ps = next_prob[i][a_i]
                    q = 0.0
                    for j, p in zip(ns, ps):
                        if task.reward_mode == "start_state":
                            r, done = reward_start_state(env, task, s), False
                        elif task.reward_mode == "intentional_entry":
                            # step cost + possible store reward if deterministic match
                            r, done = reward_intentional_entry(env, task, s, a, S[j]), False
                        else:
                            r, done = reward_terminal_once(env, task, s, S[j])
                        nxt = 0.0 if (terminal and done) else V[j]
                        q += p * (r + task.gamma * nxt)
                    tv = tie_val[i, a_i]
                    if (q > best_q + eps) or (abs(q - best_q) <= eps and tv < best_t):
                        best_q, best_t = q, tv
                newV = best_q
            d = abs(newV - V[i]); V[i] = newV
            if d > delta: delta = d
        if delta < theta: break

    # Greedy policy
    pi: Dict[State, Action] = {}
    for i, s in enumerate(S):
        if terminal and is_store(env, s):
            pi[s] = "Stay"; continue
        best_a, best_q, best_t = "Stay", -1e18, +1e18
        for a_i, a in enumerate(ACTIONS):
            ns = next_idx[i][a_i]; ps = next_prob[i][a_i]
            q = 0.0
            for j, p in zip(ns, ps):
                if task.reward_mode == "start_state":
                    r, done = reward_start_state(env, task, s), False
                elif task.reward_mode == "intentional_entry":
                    r, done = reward_intentional_entry(env, task, s, a, S[j]), False
                else:
                    r, done = reward_terminal_once(env, task, s, S[j])
                nxt = 0.0 if (terminal and done) else V[j]
                q += p * (r + task.gamma * nxt)
            tv = tie_val[i, a_i]
            if (q > best_q + 1e-12) or (abs(q - best_q) <= 1e-12 and tv < best_t):
                best_a, best_q, best_t = a, q, tv
        pi[s] = best_a
    return V, pi, S

# --------------------------- Multi-agent (Week 3) ---------------------------

@dataclass
class MultiAgentTaskSpec(TaskSpec):
    RC: float = -5.0  # crash penalty per agent when multiple agents share a cell after the step

JointState = Tuple[State, ...]
JointAction = Tuple[Action, ...]

# ---------- Fast n=2 exact VI with vectorized backups ----------
class _Precomp1:
    """Single-agent caches for fast n=2 joint planning."""
    def __init__(self, env: IceCreamGridworld):
        self.env = env
        self.S = enumerate_states(env); self.nS = len(self.S)
        self.idx = {s: i for i, s in enumerate(self.S)}
        nS, nA = self.nS, len(ACTIONS)

        self.on_road = np.array([on_road(env, s) for s in self.S], dtype=bool)
        self.store_code = np.zeros(nS, dtype=np.int8)  # 0 none, 1 RD, 2 RS
        for i, s in enumerate(self.S):
            t = is_store(env, s)
            self.store_code[i] = 1 if t == "RD" else 2 if t == "RS" else 0

        # transitions
        self.next_idx = [[None]*nA for _ in range(nS)]
        self.next_prob = [[None]*nA for _ in range(nS)]
        self.det_next_idx = np.zeros((nS, nA), dtype=int)
        self.mass_to_det = np.zeros((nS, nA), dtype=float)  # P(s' == deterministic next)
        self.tie_val = np.zeros((nS, nA), dtype=float)

        for i, s in enumerate(self.S):
            for a_i, a in enumerate(ACTIONS):
                probs = env.transition_probs(s, a)
                ns = np.array([self.idx[ss] for ss in probs.keys()], dtype=int)
                ps = np.array(list(probs.values()), dtype=float)
                self.next_idx[i][a_i] = ns
                self.next_prob[i][a_i] = ps

                det = env._apply_delta(s, DELTA[a])
                d_idx = self.idx[det]
                self.det_next_idx[i, a_i] = d_idx
                # mass to the deterministic next state
                m = 0.0
                for jj, p in zip(ns, ps):
                    if jj == d_idx: m += p
                self.mass_to_det[i, a_i] = m

                # expected nearest-store distance
                d = np.array([nearest_store_distance(env, ss) for ss in probs.keys()], dtype=float)
                self.tie_val[i, a_i] = float((ps * d).sum())

def value_iteration_multi2_fast(env: IceCreamGridworld,
                                task: MultiAgentTaskSpec,
                                theta: float = 1e-6,
                                max_iter: int = 20000):
    """
    Exact VI for two agents with vectorized backups and Gauss–Seidel updates.
    Preserves reward modes and crash semantics from Week 3.
    """
    pc = _Precomp1(env)
    nS = pc.nS; nA = len(ACTIONS)
    gamma = task.gamma
    # joint state index mapping: i = i1 * nS + i2
    def pair2i(i1, i2): return i1 * nS + i2
    def i2pair(i): return divmod(i, nS)

    # static per-state features
    road_cost = (task.RW * pc.on_road.astype(float))
    store_reward_vec = np.zeros(nS, dtype=float)
    store_reward_vec[pc.store_code == 1] = task.RD
    store_reward_vec[pc.store_code == 2] = task.RS
    store_is_terminal = (pc.store_code != 0)

    N = nS * nS
    V = np.zeros(N, dtype=float)

    # Gauss–Seidel sweep
    eps = 1e-12
    for _ in range(max_iter):
        delta = 0.0
        V_mat = V.reshape(nS, nS)  # view for fast submatrix access

        for I in range(N):
            i1, i2 = i2pair(I)

            # terminal once: if any agent already in store, V=0
            if task.reward_mode == "terminal_once" and (store_is_terminal[i1] or store_is_terminal[i2]):
                newV = 0.0
                d = abs(newV - V[I]); V[I] = newV
                if d > delta: delta = d
                continue

            best_q, best_t = -1e18, +1e18
            # precompute per-step base costs that do not depend on next
            if task.reward_mode in ("start_state", "intentional_entry"):
                step_cost_sum = road_cost[i1] + road_cost[i2]
                base_store_now = store_reward_vec[i1] + store_reward_vec[i2] if task.reward_mode == "start_state" else 0.0
            else:
                step_cost_sum = road_cost[i1] + road_cost[i2]  # used only if not terminal next
                base_store_now = 0.0

            for a1_i in range(nA):
                ns1 = pc.next_idx[i1][a1_i]; ps1 = pc.next_prob[i1][a1_i]
                tie1 = pc.tie_val[i1, a1_i]
                d1 = pc.det_next_idx[i1, a1_i]
                m1 = pc.mass_to_det[i1, a1_i]
                # store reward if deterministic next is a store
                r1_det = store_reward_vec[d1]

                for a2_i in range(nA):
                    ns2 = pc.next_idx[i2][a2_i]; ps2 = pc.next_prob[i2][a2_i]
                    tie2 = pc.tie_val[i2, a2_i]
                    d2 = pc.det_next_idx[i2, a2_i]
                    m2 = pc.mass_to_det[i2, a2_i]
                    r2_det = store_reward_vec[d2]

                    # V expectation: p1^T V_sub p2
                    V_sub = V_mat[np.ix_(ns1, ns2)]
                    expV = float(ps1 @ V_sub @ ps2)

                    # tie-break
                    tv = tie1 + tie2

                    if task.reward_mode == "start_state":
                        # start-state reward + expected next value; no crash term per original Week 3 code
                        q = base_store_now + step_cost_sum + gamma * expV

                    elif task.reward_mode == "intentional_entry":
                        # step costs now + expected store if actual next equals deterministic next + expected crash penalty + gamma * E[V]
                        # expected store: m1*r1_det + m2*r2_det
                        exp_store = m1 * r1_det + m2 * r2_det
                        # expected collision: both agents counted if they land in same cell
                        # sum_k P1(k)*P2(k), but only over shared support
                        # do small intersection since supports are tiny (<=5)
                        exp_coll_share = 0.0
                        # map smaller support first
                        if len(ns1) <= len(ns2):
                            dmap = {jj: p for jj, p in zip(ns2, ps2)}
                            for j1, p1 in zip(ns1, ps1):
                                p2 = dmap.get(j1, 0.0)
                                if p2: exp_coll_share += p1 * p2
                        else:
                            dmap = {jj: p for jj, p in zip(ns1, ps1)}
                            for j2, p2 in zip(ns2, ps2):
                                p1 = dmap.get(j2, 0.0)
                                if p1: exp_coll_share += p1 * p2
                        exp_crash = task.RC * (2.0 * exp_coll_share)
                        q = step_cost_sum + exp_store + exp_crash + gamma * expV

                    else:  # terminal_once
                        # Build terminal mask and reward matrix quickly
                        term1 = store_is_terminal[ns1]           # shape (k1,)
                        term2 = store_is_terminal[ns2]           # shape (k2,)
                        termM = term1[:, None] | term2[None, :]  # shape (k1,k2)
                        r_store1 = store_reward_vec[ns1]         # (k1,)
                        r_store2 = store_reward_vec[ns2]         # (k2,)
                        R_store = r_store1[:, None] + r_store2[None, :]
                        # step cost applies only if not terminal
                        R_step = step_cost_sum
                        # crash penalty on equal cells
                        eqM = (ns1[:, None] == ns2[None, :])
                        R_crash = task.RC * (2.0 * eqM.astype(float))
                        # total immediate reward matrix
                        R = np.where(termM, R_store, R_step) + R_crash
                        # next V zeroed on terminal pairs
                        V_eff = V_sub * (~termM)
                        q = float(ps1 @ (R + gamma * V_eff) @ ps2)

                    if (q > best_q + eps) or (abs(q - best_q) <= eps and tv < best_t):
                        best_q, best_t = q, tv

            newV = best_q
            d = abs(newV - V[I]); V[I] = newV
            if d > delta: delta = d

        if delta < theta:
            break

    # Greedy joint policy
    pi = {}
    V_mat = V.reshape(nS, nS)
    for I in range(nS * nS):
        i1, i2 = i2pair(I)
        if task.reward_mode == "terminal_once" and (store_is_terminal[i1] or store_is_terminal[i2]):
            pi[(pc.S[i1], pc.S[i2])] = ("Stay", "Stay"); continue

        best, best_t, best_a = -1e18, +1e18, (0, 0)

        step_cost_sum = road_cost[i1] + road_cost[i2] if task.reward_mode != "terminal_once" else road_cost[i1] + road_cost[i2]
        base_store_now = store_reward_vec[i1] + store_reward_vec[i2] if task.reward_mode == "start_state" else 0.0

        for a1_i in range(nA):
            ns1 = pc.next_idx[i1][a1_i]; ps1 = pc.next_prob[i1][a1_i]
            tie1 = pc.tie_val[i1, a1_i]; d1 = pc.det_next_idx[i1, a1_i]
            m1 = pc.mass_to_det[i1, a1_i]; r1_det = store_reward_vec[d1]
            for a2_i in range(nA):
                ns2 = pc.next_idx[i2][a2_i]; ps2 = pc.next_prob[i2][a2_i]
                tie2 = pc.tie_val[i2, a2_i]; d2 = pc.det_next_idx[i2, a2_i]
                m2 = pc.mass_to_det[i2, a2_i]; r2_det = store_reward_vec[d2]
                tv = tie1 + tie2
                V_sub = V_mat[np.ix_(ns1, ns2)]
                if task.reward_mode == "start_state":
                    q = base_store_now + step_cost_sum + gamma * float(ps1 @ V_sub @ ps2)
                elif task.reward_mode == "intentional_entry":
                    expV = float(ps1 @ V_sub @ ps2)
                    # crash expectation
                    exp_coll_share = 0.0
                    if len(ns1) <= len(ns2):
                        dmap = {jj: p for jj, p in zip(ns2, ps2)}
                        for j1, p1 in zip(ns1, ps1):
                            p2 = dmap.get(j1, 0.0)
                            if p2: exp_coll_share += p1 * p2
                    else:
                        dmap = {jj: p for jj, p in zip(ns1, ps1)}
                        for j2, p2 in zip(ns2, ps2):
                            p1 = dmap.get(j2, 0.0)
                            if p1: exp_coll_share += p1 * p2
                    exp_crash = task.RC * (2.0 * exp_coll_share)
                    q = step_cost_sum + (m1 * r1_det + m2 * r2_det) + exp_crash + gamma * expV
                else:
                    term1 = store_is_terminal[ns1]; term2 = store_is_terminal[ns2]
                    termM = term1[:, None] | term2[None, :]
                    r_store1 = store_reward_vec[ns1]; r_store2 = store_reward_vec[ns2]
                    R_store = r_store1[:, None] + r_store2[None, :]
                    eqM = (ns1[:, None] == ns2[None, :])
                    R_crash = task.RC * (2.0 * eqM.astype(float))
                    V_eff = V_sub * (~termM)
                    R = np.where(termM, R_store, step_cost_sum) + R_crash
                    q = float(ps1 @ (R + gamma * V_eff) @ ps2)

                if (q > best + 1e-12) or (abs(q - best) <= 1e-12 and tv < best_t):
                    best, best_t, best_a = q, tv, (a1_i, a2_i)

        pi[(pc.S[i1], pc.S[i2])] = (I2A[best_a[0]], I2A[best_a[1]])
    return V, pi, [ (s1, s2) for s1 in pc.S for s2 in pc.S ]

# ---------- Lightweight approximate solver (same as before, smaller defaults) ----------

def default_joint_features(env: IceCreamGridworld) -> Callable[[JointState], np.ndarray]:
    RD, RS = env.spec.RD, env.spec.RS
    roads = env.spec.roads
    def phi(s: JointState) -> np.ndarray:
        k = len(s)
        d_near = [min(manhattan(si, RD), manhattan(si, RS)) for si in s]
        d_RD   = [manhattan(si, RD) for si in s]
        d_RS   = [manhattan(si, RS) for si in s]
        pair_d = []
        for i in range(k):
            for j in range(i+1, k):
                pair_d.append(manhattan(s[i], s[j]))
        on_road_count = sum(1 for si in s if si in roads)
        counts: Dict[State, int] = {}
        for si in s: counts[si] = counts.get(si, 0) + 1
        collided_now = sum(c for c in counts.values() if c > 1)
        feats = np.array([
            1.0,
            float(sum(d_near)),
            float(min(d_RD)),
            float(min(d_RS)),
            float(sum(pair_d) if pair_d else 0.0),
            float(on_road_count),
            float(collided_now),
        ], dtype=float)
        return feats / (1.0 + np.linalg.norm(feats))
    return phi

def fitted_value_iteration_linear(env: IceCreamGridworld,
                                  task: MultiAgentTaskSpec,
                                  n_agents: int = 3,
                                  iterations: int = 12,
                                  samples_per_iter: int = 1200,
                                  feature_fn: Optional[Callable[[JointState], np.ndarray]] = None,
                                  seed: Optional[int] = 0):
    rng = np.random.default_rng(seed)
    if feature_fn is None:
        feature_fn = default_joint_features(env)
    S1 = enumerate_states(env)

    def sample_joint_state() -> JointState:
        return tuple(S1[rng.integers(0, len(S1))] for _ in range(n_agents))

    d = len(feature_fn(tuple([S1[0]] * n_agents)))
    w = np.zeros(d, dtype=float)

    def joint_transition_probs(s: JointState, a: JointAction) -> Dict[JointState, float]:
        per_agent = [env.transition_probs(s[i], a[i]) for i in range(len(s))]
        probs: Dict[JointState, float] = {}
        for combo in itertools.product(*[list(p.items()) for p in per_agent]):
            next_positions = tuple(item[0] for item in combo)
            p = 1.0
            for item in combo: p *= item[1]
            probs[next_positions] = probs.get(next_positions, 0.0) + p
        return probs

    def backup(s: JointState) -> float:
        best = -1e18; best_t = +1e18
        for a in itertools.product(ACTIONS, repeat=n_agents):
            probs = joint_transition_probs(s, a)
            q = 0.0
            for s_next, p in probs.items():
                # minimal faithful rewards to Week 3
                step_cost = sum(task.RW for si in s if on_road(env, si)) if task.reward_mode != "terminal_once" else sum(task.RW for si in s)
                if task.reward_mode == "intentional_entry":
                    # deterministic success bonus if next equals deterministic for that agent
                    bonus = 0.0
                    for i in range(n_agents):
                        det = env._apply_delta(s[i], DELTA[a[i]])
                        if s_next[i] == det:
                            t = is_store(env, s_next[i])
                            if t == "RD": bonus += task.RD
                            elif t == "RS": bonus += task.RS
                    # crash penalty
                    counts: Dict[State, int] = {}
                    for si in s_next: counts[si] = counts.get(si, 0) + 1
                    collided_agents = sum(c for c in counts.values() if c > 1)
                    r = step_cost + bonus + task.RC * collided_agents
                    done = False
                elif task.reward_mode == "start_state":
                    r = step_cost + sum((task.RD if is_store(env, si) == "RD" else task.RS if is_store(env, si) == "RS" else 0.0) for si in s)
                    done = False
                else:
                    done = any(is_store(env, si) for si in s_next)
                    r = sum((task.RD if is_store(env, si) == "RD" else task.RS if is_store(env, si) == "RS" else 0.0) for si in s_next) if done else step_cost
                v_next = 0.0 if done else float(np.dot(w, feature_fn(s_next)))
                q += p * (r + task.gamma * v_next)
            # simple distance tie-break
            tv = sum(min(manhattan(si, env.spec.RD), manhattan(si, env.spec.RS)) for si in s)
            if (q > best) or (abs(q - best) <= 1e-12 and tv < best_t):
                best, best_t = q, tv
        return best

    for _ in range(iterations):
        X = []; y = []
        for _ in range(samples_per_iter):
            s = sample_joint_state()
            X.append(feature_fn(s)); y.append(backup(s))
        X = np.stack(X, axis=0); y = np.array(y, dtype=float)
        lam = 1e-3
        XT_X = X.T @ X + lam * np.eye(X.shape[1])
        XT_y = X.T @ y
        w = np.linalg.solve(XT_X, XT_y)

    def greedy_pi(s: JointState) -> JointAction:
        best, best_t, best_a = -1e18, +1e18, None
        for a in itertools.product(ACTIONS, repeat=n_agents):
            probs = joint_transition_probs(s, a)
            q = 0.0
            for s_next, p in probs.items():
                step_cost = sum(task.RW for si in s if on_road(env, si))
                done = any(is_store(env, si) for si in s_next) if task.reward_mode == "terminal_once" else False
                v_next = 0.0 if done else float(np.dot(w, feature_fn(s_next)))
                q += p * (step_cost + task.gamma * v_next)
            tv = sum(min(manhattan(si, env.spec.RD), manhattan(si, env.spec.RS)) for si in s)
            if (q > best) or (abs(q - best) <= 1e-12 and tv < best_t):
                best, best_t, best_a = q, tv, a
        return best_a if best_a is not None else tuple(["Stay"] * n_agents)

    return w, feature_fn, greedy_pi

# ----------------------- Viz helpers -----------------------

ARROW = {"Up": "↑", "Down": "↓", "Left": "←", "Right": "→", "Stay": "•"}

def plot_policy(env, pi, S, title="Policy"):
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    env.render(ax=ax, title=title)
    for s, a in pi.items():
        x, y = s
        ax.text(x + 0.5, y + 0.5, ARROW[a], ha="center", va="center", fontsize=16)
    plt.show()

def render_multi(env: IceCreamGridworld, s: JointState, ax=None, title=""):
    close_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        close_ax = True
    env.render(ax=ax, title=title)
    for i, pos in enumerate(s):
        ax.add_patch(patches.Circle((pos[0] + 0.5, pos[1] + 0.5), 0.24, fill=True, alpha=0.7))
        ax.text(pos[0] + 0.5, pos[1] + 0.5, str(i), ha="center", va="center", color="white", fontsize=12)
    if close_ax:
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

def rollout_multi(env: IceCreamGridworld, pi_fn: Callable[[JointState], JointAction],
                  start: Optional[JointState], steps: int = 20, seed: Optional[int] = 0):
    rng = np.random.default_rng(seed)
    env.rng = rng
    if start is None:
        raise ValueError("Please provide a starting JointState for multi-agent rollout.")
    traj = [start]; s = start
    for _ in range(steps):
        a = pi_fn(s)
        # simple product sampler for two agents
        probs1 = env.transition_probs(s[0], a[0]); probs2 = env.transition_probs(s[1], a[1])
        next1 = list(probs1.keys()); p1 = np.array(list(probs1.values())); p1 /= p1.sum()
        next2 = list(probs2.keys()); p2 = np.array(list(probs2.values())); p2 /= p2.sum()
        s1 = next1[rng.choice(len(next1), p=p1)]
        s2 = next2[rng.choice(len(next2), p=p2)]
        s = (s1, s2); traj.append(s)
    return traj

# --------------------------- Run as script ---------------------------

if __name__ == "__main__":
    # --- Map / Environment ---
    spec = GridSpec(
        width=5, height=5,
        obstacles={(1, 1), (1, 3), (3, 1), (3, 3)},
        RD=(4, 4), RS=(0, 0),
        start=(1, 2),
        p_error=0.2, seed=42,
        roads={(0, 2), (1, 2), (2, 2)}  # red road cells (passable, penalized)
    )
    env = IceCreamGridworld(spec)

    # For visibility, show which backend you're on (optional)
    import matplotlib
    print("Matplotlib backend:", matplotlib.get_backend())

    # --- 1) Single-agent planning + policy plot ---
    task_sa = TaskSpec(RD=10.0, RS=10.0, RW=-10.0, gamma=0.95, reward_mode="start_state")
    V_sa, pi_sa, S_sa = value_iteration(env, task_sa)
    print(f"[Single-agent] |S|={len(S_sa)}, V∈[{V_sa.min():.2f},{V_sa.max():.2f}]")
    plot_policy(env, pi_sa, S_sa, title="Single-agent policy (roads penalized)")

    # --- 2) Week 3 exact n=2 (fast) + static snapshot ---
    task_ma = MultiAgentTaskSpec(RD=10.0, RS=10.0, RW=-1.0, RC=-5.0, gamma=0.95, reward_mode="intentional_entry")
    V2, pi2, S2 = value_iteration_multi2_fast(env, task_ma, theta=5e-6, max_iter=2000)
    print(f"[Exact n=2 fast] joint={len(S2)}, V∈[{V2.min():.2f},{V2.max():.2f}]")

    js = (spec.start, (2, 2))
    if js not in pi2:
        js = S2[0]
    print(f"[Exact n=2 fast] Greedy at {js}: {pi2[js]}")

    # Static board with two agents at starting joint state
    render_multi(env, js, title="n=2 start (exact VI)")

    # --- 3) Quick n=2 greedy rollout animation ---
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    traj2 = rollout_multi(env, lambda s: pi2[s], start=js, steps=12, seed=0)
    for t, s in enumerate(traj2):
        # render_multi clears and redraws the grid + agents on the same axes
        render_multi(env, s, ax=ax, title=f"n=2 greedy rollout t={t}")
        plt.pause(0.25)  # small delay so you can see the motion

    # --- 4) Approximate n=3: start/end snapshots ---
    w, phi, pi_greedy = fitted_value_iteration_linear(
        env, task_ma, n_agents=3, iterations=10, samples_per_iter=900, seed=0
    )
    start3 = (spec.start, (0, 1), (4, 0))
    traj3 = rollout_multi(env, pi_greedy, start=start3, steps=8, seed=1)
    render_multi(env, traj3[0], title="n=3 rollout start (approx)")
    render_multi(env, traj3[-1], title="n=3 rollout end (approx)")

    # Keep all figures open until you close them
    plt.show()