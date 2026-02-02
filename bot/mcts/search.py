from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import random

from bot.mcts.shadow_state import ShadowState, Action
from bot.mcts.eval import evaluate_state

@dataclass
class MCTSConfig:
    num_simulations: int = 300
    max_depth: int = 4
    c_puct: float = 1.6

    # Exploration noise at root (optional)
    dirichlet_alpha: float = 0.0
    dirichlet_eps: float = 0.0

    # Action selection at root
    temperature: float = 0.0  # 0 => argmax visits

    seed: Optional[int] = None

class Node:
    __slots__ = ("state", "parent", "prior", "children", "N", "W")

    def __init__(self, state: ShadowState, parent: Optional["Node"] = None, prior: float = 1.0):
        self.state: ShadowState = state
        self.parent: Optional["Node"] = parent
        self.prior: float = float(prior)  # P(a|s) from heuristics
        self.children: Dict[Action, "Node"] = {}
        self.N: int = 0
        self.W: float = 0.0  # total value backed up

    @property
    def Q(self) -> float:
        return 0.0 if self.N == 0 else (self.W / self.N)

    def is_expanded(self) -> bool:
        return len(self.children) > 0

def softmax(scores: List[float], tau: float = 12.0) -> List[float]:
    if not scores:
        return []
    tau = max(float(tau), 1e-9)
    m = max(scores)
    exps = [math.exp((s - m) / tau) for s in scores]
    z = sum(exps)
    if z <= 0.0 or not math.isfinite(z):
        return [1.0 / len(scores)] * len(scores)
    return [e / z for e in exps]


def action_priors(state: ShadowState, actions: List[Action]) -> Dict[Action, float]:
    """
    Convert heuristic scores into priors P(a|s) for PUCT.
    """
    scores: List[float] = []
    with state._patched_status():
        for kind, obj in actions:
            if kind == "move":
                scores.append(float(state.score_move_fn(obj, state.battle, state.ctx_me)))
            else:
                scores.append(float(state.score_switch_fn(obj, state.battle, state.ctx_me)))

    probs = softmax(scores, tau=12.0)

    priors: Dict[Action, float] = {}
    for a, p in zip(actions, probs):
        priors[a] = float(p)
    return priors


def add_dirichlet_noise(priors: Dict[Action, float], alpha: float, eps: float, rng: random.Random) -> Dict[Action, float]:
    if eps <= 0.0 or alpha <= 0.0 or len(priors) <= 1:
        return priors

    actions = list(priors.keys())
    noise_raw = [rng.gammavariate(alpha, 1.0) for _ in actions]
    z = sum(noise_raw) or 1.0
    noise = [x / z for x in noise_raw]

    out: Dict[Action, float] = {}
    for a, n in zip(actions, noise):
        out[a] = (1.0 - eps) * priors[a] + eps * n

    s = sum(out.values()) or 1.0
    for a in out:
        out[a] /= s
    return out

def select_child(node: Node, c_puct: float) -> Tuple[Action, Node]:
    """
    Pick child that maximizes:
      Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
    """
    sqrt_N = math.sqrt(max(1, node.N))
    best_a: Optional[Action] = None
    best_u = -1e18
    best_child: Optional[Node] = None

    for a, child in node.children.items():
        q = child.Q
        u = q + c_puct * child.prior * (sqrt_N / (1 + child.N))
        if u > best_u:
            best_u = u
            best_a = a
            best_child = child

    return best_a, best_child  # type: ignore

def expand(node: Node, rng: random.Random, cfg: MCTSConfig, is_root: bool) -> None:
    if node.state.is_terminal():
        return

    actions = node.state.legal_actions()
    if not actions:
        return

    priors = action_priors(node.state, actions)
    if is_root:
        priors = add_dirichlet_noise(priors, cfg.dirichlet_alpha, cfg.dirichlet_eps, rng)

    for a in actions:
        if a in node.children:
            continue
        next_state = node.state.step(a)
        node.children[a] = Node(next_state, parent=node, prior=priors.get(a, 1.0 / len(actions)))

def evaluate_leaf(node: Node) -> float:
    """
    Leaf evaluation from OUR perspective, bounded ~[-1,1].
    """
    return float(evaluate_state(node.state))


def backup(node: Node, value: float) -> None:
    cur: Optional[Node] = node
    while cur is not None:
        cur.N += 1
        cur.W += value
        cur = cur.parent

def action_name(a: Action) -> Tuple[str, str]:
    kind, obj = a
    if kind == "move":
        return "move", str(getattr(obj, "id", "") or getattr(obj, "name", "") or "move")
    return "switch", str(getattr(obj, "species", "") or getattr(obj, "name", "") or "switch")

def search(
    *,
    battle: Any,
    ctx_me: Any,
    ctx_opp: Any,
    score_move_fn,
    score_switch_fn,
    dmg_fn,
    cfg: MCTSConfig,
    opp_tau: float = 8.0,
    status_threshold: float = 0.30,
    allow_switches: bool = True,
    return_stats: bool = False,
) -> Union[Action, Tuple[Action, Dict[str, Any]]]:
    """
    Runs PUCT MCTS and returns best root action.
    If return_stats=True, also returns a dict with root child visits/Q/prior.

    allow_switches=False => only expands root with moves (still allows pivot-triggered switches inside rollouts).
    """
    rng = random.Random(cfg.seed)

    root_state = ShadowState.from_battle(
        battle=battle,
        ctx_me=ctx_me,
        ctx_opp=ctx_opp,
        score_move_fn=score_move_fn,
        score_switch_fn=score_switch_fn,
        dmg_fn=dmg_fn,
        opp_tau=opp_tau,
        status_threshold=status_threshold,
    )
    root = Node(root_state, parent=None, prior=1.0)

    # Expand root once
    expand(root, rng, cfg, is_root=True)

    # Optional: restrict root to moves only
    if not allow_switches and root.children:
        root.children = {a: n for a, n in root.children.items() if a[0] == "move"}
        if not root.children:
            # If we removed everything, just re-expand with legal moves only by filtering actions
            # simplest fallback: return first available move
            actions = root_state.legal_actions()
            moves = [a for a in actions if a[0] == "move"]
            if moves:
                picked = moves[0]
                return (picked, {"top": [], "sims": 0}) if return_stats else picked

    if not root.children:
        picked = ("move", None)
        return (picked, {"top": [], "sims": 0}) if return_stats else picked

    # Simulations
    for _ in range(int(cfg.num_simulations)):
        node = root
        depth = 0

        # Selection
        while node.is_expanded() and not node.state.is_terminal() and depth < cfg.max_depth:
            _, node = select_child(node, cfg.c_puct)
            depth += 1

        # Expansion (only if we stopped on an unexpanded node)
        if not node.state.is_terminal() and depth < cfg.max_depth and not node.is_expanded():
            expand(node, rng, cfg, is_root=False)

        # Evaluation
        value = evaluate_leaf(node)

        # Backup
        backup(node, value)

    # Build root stats
    actions = list(root.children.keys())
    rows: List[Dict[str, Any]] = []
    for a in actions:
        child = root.children[a]
        kind, name = action_name(a)
        rows.append(
            {
                "kind": kind,
                "name": name,
                "visits": int(child.N),
                "q": float(child.Q),
                "prior": float(child.prior),
            }
        )
    rows.sort(key=lambda d: (d["visits"], d["q"]), reverse=True)

    # Pick best action from root children
    if cfg.temperature and cfg.temperature > 1e-9:
        # sample proportional to exp(visits / T)
        visits = [root.children[a].N for a in actions]
        m = max(visits)
        weights = [math.exp((v - m) / float(cfg.temperature)) for v in visits]
        z = sum(weights) or 1.0
        r = rng.random() * z
        acc = 0.0
        picked = actions[-1]
        for a, w in zip(actions, weights):
            acc += w
            if acc >= r:
                picked = a
                break
    else:
        picked = max(actions, key=lambda a: root.children[a].N)

    if return_stats:
        return picked, {"top": rows[:10], "sims": int(cfg.num_simulations)}
    return picked

def mcts_pick_action(
    *,
    battle: Any,
    ctx: Any,
    ctx_opp: Any,
    score_move_fn,
    score_switch_fn,
    dmg_fn,
    iters: int,
    max_depth: int = 4,
    include_switches: bool = True,
    opp_tau: float = 8.0,
    status_threshold: float = 0.30,
) -> Tuple[Optional[Action], Optional[Dict[str, Any]]]:
    """
    Wrapper that matches your old pattern: returns (picked, stats).
    """
    cfg = MCTSConfig(
        num_simulations=int(iters),
        max_depth=int(max_depth),
        c_puct=1.6,
        dirichlet_alpha=0.0,
        dirichlet_eps=0.0,
        temperature=0.0,
        seed=None,
    )

    picked, stats = search(
        battle=battle,
        ctx_me=ctx,
        ctx_opp=ctx_opp,
        score_move_fn=score_move_fn,
        score_switch_fn=score_switch_fn,
        dmg_fn=dmg_fn,
        cfg=cfg,
        opp_tau=opp_tau,
        status_threshold=status_threshold,
        allow_switches=bool(include_switches),
        return_stats=True,
    )
    return picked, stats
