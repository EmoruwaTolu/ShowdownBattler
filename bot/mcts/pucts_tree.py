from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

Action = Tuple[str, Any]  # ("move", Move) or ("switch", Pokemon)

class SearchState(Protocol):
    def legal_actions(self) -> List[Action]: ...
    def is_terminal(self) -> bool: ...

@dataclass(frozen=True)
class BattleState:
    """
    Minimal wrapper around a poke-env battle + your EvalContext.

    For selection/traversal, we only need:
      - legal_actions()
      - is_terminal()

    Later (expansion), we will add:
      - step(action) -> BattleState
    """
    battle: Any
    ctx: Any  # EvalContext (your project type)

    def legal_actions(self) -> List[Action]:
        actions: List[Action] = []

        # Moves currently available
        for m in getattr(self.battle, "available_moves", []) or []:
            actions.append(("move", m))

        # Switches currently available
        for p in getattr(self.battle, "available_switches", []) or []:
            actions.append(("switch", p))

        return actions

    def is_terminal(self) -> bool:
        # Common poke-env flags
        return bool(getattr(self.battle, "won", False) or getattr(self.battle, "lost", False))

class HeuristicPrior:
    """
    Converts your heuristic(s) into a real-valued action score.

    - Moves use your score_move(move, battle, ctx)
    - Switches can optionally use a switch scoring function
    """
    def __init__(self, *, move_score_fn, switch_score_fn=None):
        """
        move_score_fn: function(move, battle, ctx) -> float
        switch_score_fn: optional function(pokemon, battle, ctx) -> float
        """
        self.move_score_fn = move_score_fn
        self.switch_score_fn = switch_score_fn

    def action_score(self, state: BattleState, action: Action) -> float:
        kind, obj = action
        battle, ctx = state.battle, state.ctx

        if kind == "move":
            return float(self.move_score_fn(obj, battle, ctx))

        if kind == "switch":
            if self.switch_score_fn is not None:
                return float(self.switch_score_fn(obj, battle, ctx))
            # Safe default: allow exploration but slightly discourage vs moves
            return -5.0

        return -100.0


def softmax_priors(actions: List[Action], scores: List[float], *, tau: float = 12.0, min_prob: float = 1e-6) -> Dict[Action, float]:
    """
    Convert arbitrary heuristic scores into a probability distribution.

    temperature:
      - larger -> flatter distribution (more exploration)
      - smaller -> peakier distribution (trusts heuristic more)

    min_prob:
      - prevents exact zeros (useful for exploration + numeric safety)
    """
    assert len(actions) == len(scores) and len(actions) > 0

    m = max(scores)
    exps: List[float] = []

    for s in scores:
        z = (s - m) / max(1e-9, tau)
        z = max(-50.0, min(50.0, z))  # clamp for stability
        exps.append(math.exp(z))

    total = sum(exps)
    if total <= 0.0 or not math.isfinite(total):
        p = 1.0 / len(actions)
        return {a: p for a in actions}

    priors = {a: e / total for a, e in zip(actions, exps)}

    if min_prob > 0.0:
        priors = {a: max(min_prob, p) for a, p in priors.items()}
        z = sum(priors.values())
        priors = {a: p / z for a, p in priors.items()}

    return priors


@dataclass
class Node:
    """
    A node in the MCTS tree.

    Stores:
      - visit count N, total value W, mean Q
      - children
      - untried actions (for expansion later)
      - P(a): policy prior distribution derived from heuristic action scores
    """
    state: BattleState
    prior: HeuristicPrior
    parent: Optional["Node"] = None
    parent_action: Optional[Action] = None

    # MCTS stats
    N: int = 0
    W: float = 0.0

    # Tree structure
    children: Dict[Action, "Node"] = field(default_factory=dict)
    untried_actions: List[Action] = field(default_factory=list)

    # Policy priors for actions at this node
    P: Dict[Action, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.untried_actions = list(self.state.legal_actions())

        if self.untried_actions:
            scores = [self.prior.action_score(self.state, a) for a in self.untried_actions]
            self.P = softmax_priors(self.untried_actions, scores, tau=12.0)
        else:
            self.P = {}

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


def select_child_puct(node: Node, *, c: float = 1.8) -> Node:
    """
    Select a child according to PUCT:

      PUCT(a) = Q(a) + c * P(a) * sqrt(N_parent) / (1 + N(a))

      - Uses node.P[action] as a heuristic-derived prior.
      - Q is learned from rollouts/backprop (later).
    """
    if not node.children:
        raise ValueError("select_child_puct called on a node with no children")

    best_score = -1e18
    best: List[Node] = []

    sqrt_parent = math.sqrt(max(1, node.N))

    for action, child in node.children.items():
        p = node.P.get(action, 1e-6)
        u = c * p * (sqrt_parent / (1 + child.N))
        score = child.Q + u

        if score > best_score:
            best_score = score
            best = [child]
        elif score == best_score:
            best.append(child)

    return random.choice(best)

def tree_traverse(root: Node, *, c: float = 1.8) -> Tuple[Node, List[Node]]:
    """
    Selection / Traversal:

    Starting at root:
      - If current node is terminal -> stop.
      - If current node has untried actions -> stop (ready for expansion).
      - Else choose a child by PUCT and continue.

    Returns:
      (leaf_node, path)
        - leaf_node is either terminal or expandable
        - path is the sequence of nodes visited from root to leaf_node
    """
    node = root
    path: List[Node] = [node]

    while not node.state.is_terminal():
        if node.untried_actions:
            break
        node = select_child_puct(node, c=c)
        path.append(node)

    return node, path
