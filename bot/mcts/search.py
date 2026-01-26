from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from bot.model.opponent_model import OpponentBelief, determinize_opponent
from bot.scoring.damage_score import estimate_damage_fraction
from bot.scoring.helpers import hp_frac
from bot.scoring.move_score import score_move
from bot.scoring.switch_score import score_switch
from poke_env.battle import Status

Action = Tuple[str, Any]  # ("move" or "switch", object)

@dataclass
class ShadowState:
    my_hp: float
    opp_hp: float
    my_status: Optional[Status] = None
    opp_status: Optional[Status] = None
    opp_dpt: float = 0.25


def eval_state(s: ShadowState) -> float:
    # TODO:  simple: HP advantage + mild shaping. Will replace with race-based eval later.
    return 80.0 * (s.my_hp - s.opp_hp)

def apply_my_action(s: ShadowState, battle: Any, ctx: Any, action: Action) -> ShadowState:
    kind, obj = action
    my_hp = s.my_hp
    opp_hp = s.opp_hp

    if kind == "move":
        mv = obj
        dmg = float(estimate_damage_fraction(mv, ctx.me, ctx.opp, battle))
        acc = float(getattr(mv, "accuracy", 1.0) or 1.0)
        acc = max(0.0, min(1.0, acc))
        opp_hp = max(0.0, opp_hp - dmg * acc)
        return ShadowState(my_hp=my_hp, opp_hp=opp_hp, opp_dpt=s.opp_dpt)

    # Switch modeling placeholder: small tempo cost, slightly reduced immediate threat
    if kind == "switch":
        return ShadowState(
            my_hp=max(0.0, my_hp - 0.05),
            opp_hp=opp_hp,
            opp_dpt=max(0.10, s.opp_dpt - 0.03),
        )

    return s


def apply_opp_reply(s: ShadowState, reply_kind: str) -> ShadowState:
    my_hp = s.my_hp
    opp_hp = s.opp_hp
    opp_dpt = s.opp_dpt

    if reply_kind == "attack":
        return ShadowState(my_hp=max(0.0, my_hp - opp_dpt), opp_hp=opp_hp, opp_dpt=opp_dpt)

    if reply_kind == "setup":
        return ShadowState(my_hp=my_hp, opp_hp=opp_hp, opp_dpt=min(0.85, opp_dpt + 0.15))

    return s

def opponent_replies_for_sim(world: Any) -> List[str]:
    # Default replies
    replies = ["attack"]

    # If we have a sampled role candidate and it can setup, add setup reply
    try:
        if world is not None and getattr(world, "candidate", None) is not None:
            if getattr(world.candidate, "has_setup", False):
                replies.append("setup")
    except Exception:
        pass

    return replies

@dataclass
class Node:
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    children: Dict[Action, "Node"] = field(default_factory=dict)

    def q(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


def ucb(parent_visits: int, child: Node, c_puct: float = 1.4) -> float:

    return child.q() + c_puct * child.prior * math.sqrt(parent_visits + 1) / (1 + child.visits)


def action_prior(action: Action, battle: Any, ctx: Any) -> float:
    kind, obj = action

    if kind == "move":
        s = float(score_move(obj, battle, ctx))
    else:
        s = float(score_switch(obj, battle, ctx))

    return 1.0 / (1.0 + math.exp(-0.02 * s))


def top_k_actions(actions: List[Action], battle: Any, ctx: Any, k: int) -> List[Action]:
    if k <= 0 or len(actions) <= k:
        return actions
    scored = [(a, action_prior(a, battle, ctx)) for a in actions]
    scored.sort(key=lambda t: t[1], reverse=True)
    return [a for a, _ in scored[:k]]


def expand(node: Node, actions: List[Action], battle: Any, ctx: Any) -> None:
    if node.children:
        return
    priors = [action_prior(a, battle, ctx) for a in actions]
    total = sum(priors) or 1.0
    for a, p in zip(actions, priors):
        node.children[a] = Node(prior=p / total)


def rollout_value(
    s: ShadowState,
    depth_left: int,
    world: Any,
    rng: random.Random,
    battle: Any,
    ctx: Any,
    my_actions: List[Action],
    my_k: int = 4,
) -> float:
    """
    Alternating rollout:
      - Opponent ply: min over small reply set (attack/setup)
      - My ply: max over top-k actions by prior (beam)
    depth_left counts remaining plies.
    We start at an opponent ply because mcts_pick_action applies our root action first.
    """
    if depth_left <= 0 or s.my_hp <= 0.0 or s.opp_hp <= 0.0:
        return eval_state(s)

    # Opponent chooses worst-case reply
    replies = opponent_replies_for_sim(world)
    worst = float("inf")

    for r in replies:
        s_after_opp = apply_opp_reply(s, r)

        if depth_left - 1 <= 0 or s_after_opp.my_hp <= 0.0 or s_after_opp.opp_hp <= 0.0:
            worst = min(worst, eval_state(s_after_opp))
            continue

        # Our response: choose best action among top-k
        best_next = -1e9
        for a in top_k_actions(my_actions, battle, ctx, my_k):
            s_after_me = apply_my_action(s_after_opp, battle, ctx, a)
            v = rollout_value(
                s_after_me,
                depth_left - 2,
                world,
                rng,
                battle,
                ctx,
                my_actions,
                my_k=my_k,
            )
            best_next = max(best_next, v)

        worst = min(worst, best_next)

    return worst


def mcts_pick_action(
    *,
    battle: Any,
    ctx: Any,
    belief: Optional[OpponentBelief],
    actions: List[Action],
    iters: int = 120,
    max_depth: int = 4,
    include_switches: bool = True,
    root_k: int = 6,  # prune root branching
    rollout_k: int = 4,  # prune my move branching inside rollouts
    c_puct: float = 1.4,
) -> Tuple[Optional[Action], Optional[dict]]:
    rng = random.Random()

    if not include_switches:
        actions = [a for a in actions if a[0] == "move"]

    if not actions:
        return None, None

    # Root pruning (huge for speed once switching grows)
    actions_for_root = top_k_actions(actions, battle, ctx, root_k)

    root = Node(prior=1.0)
    expand(root, actions_for_root, battle, ctx)

    # Base shadow state from actual HP fractions
    s0 = ShadowState(my_hp=float(hp_frac(ctx.me)), opp_hp=float(hp_frac(ctx.opp)))

    for _ in range(max(1, iters)):
        # determinize opponent each simulation
        world = None
        if belief is not None:
            try:
                world = determinize_opponent(belief, rng)
            except Exception:
                world = None

        # Select root action by PUCT
        best_a: Optional[Action] = None
        best_score = -1e18
        for a, child in root.children.items():
            sc = ucb(root.visits, child, c_puct=c_puct)
            if sc > best_score:
                best_score = sc
                best_a = a

        if best_a is None:
            break

        # Apply our chosen root action
        s1 = apply_my_action(s0, battle, ctx, best_a)

        # Rollout from opponent reply (alternating)
        v = rollout_value(
            s1,
            max_depth - 1,
            world,
            rng,
            battle,
            ctx,
            actions_for_root,
            my_k=rollout_k,
        )

        # Backprop (root-only tree for now)
        child = root.children[best_a]
        child.visits += 1
        child.value_sum += v
        root.visits += 1

    # Choose action by most visits
    best: Optional[Action] = None
    best_vis = -1
    top = []
    for a, child in root.children.items():
        top.append((a, child.visits, child.q()))
        if child.visits > best_vis:
            best_vis = child.visits
            best = a

    top.sort(key=lambda t: t[1], reverse=True)
    stats = {
        "top": [
            {
                "kind": a[0],
                "name": getattr(a[1], "id", str(a[1])),
                "visits": v,
                "q": q,
            }
            for a, v, q in top[:8]
        ]
    }
    return best, stats
