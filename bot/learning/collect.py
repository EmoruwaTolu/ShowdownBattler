from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class TurnRecord:
    """One decision point captured after MCTS returns."""

    # Identity
    battle_id: str
    turn: int

    # Value-function features (Phase A: predict outcome z)
    eval_terms: Dict[str, float]    # φ(s): 24 named components from evaluate_state_terms()
    eval_value: float               # weighted-sum heuristic value at root

    # Policy features (Phase B: predict visit fractions)
    # Each entry: {"kind": "move"|"switch", "name": str, "visits": int, "frac": float,
    #              "prior": float, "q": float}
    action_stats: List[Dict[str, Any]]

    # Opponent model context (useful feature for prior model)
    # Top-k predicted opponent moves with probabilities.
    # Each entry: {"name": str, "prob": float}
    opp_top_moves: List[Dict[str, float]]

    # Shannon entropy of the active opponent Pokemon's belief distribution (nats).
    # 0 = fully known set, high = lots of uncertainty.
    belief_entropy: float

    # Decision 
    picked_kind: str     # "move" or "switch"
    picked_name: str     # move id or pokemon species

    # Outcome (filled in by finish_battle)
    # +1 = win, -1 = loss, 0 = tie / unknown
    outcome: int = 0

def _belief_entropy(belief: Any) -> float:
    """Shannon entropy (nats) of an OpponentBelief distribution."""
    if belief is None:
        return 0.0
    dist = getattr(belief, "dist", None)
    if not dist:
        return 0.0
    H = 0.0
    for _, p in dist:
        p = float(p)
        if p > 1e-12:
            H -= p * math.log(p)
    return float(H)


def _opp_top_moves(
    opp_beliefs: Dict[int, Any],
    opp_active: Any,
    score_move_fn: Any,
    ctx_opp: Any,
    battle: Any,
    k: int = 5,
) -> List[Dict[str, float]]:
    """
    Collect the opponent's predicted move distribution from the belief move pool,
    weighted by role probability and heuristic score.

    Returns a list of {"name": str, "prob": float} sorted descending.
    """
    if not opp_beliefs or opp_active is None or score_move_fn is None or ctx_opp is None:
        return []

    belief = opp_beliefs.get(id(opp_active))
    if belief is None:
        return []

    dist = getattr(belief, "dist", None)
    if not dist:
        return []

    # Accumulate weighted score per move across all candidate roles
    move_weight: Dict[str, float] = {}
    for cand, role_prob in dist:
        rp = float(role_prob)
        if rp < 1e-6:
            continue
        moves = getattr(cand, "moves", set()) or set()
        for mv_id in moves:
            move_weight[mv_id] = move_weight.get(mv_id, 0.0) + rp

    if not move_weight:
        return []

    # Softmax with temperature 8 (matches choose_opp_action in search.py)
    import math as _math
    TAU = 8.0
    scores: Dict[str, float] = {}
    for mv_id, w in move_weight.items():
        scores[mv_id] = w  # use role-weighted probability as proxy score

    max_s = max(scores.values())
    exps = {mv: _math.exp((s - max_s) / TAU) for mv, s in scores.items()}
    total = sum(exps.values()) + 1e-9
    probs = {mv: e / total for mv, e in exps.items()}

    top = sorted(probs.items(), key=lambda x: -x[1])[:k]
    return [{"name": mv, "prob": round(p, 4)} for mv, p in top]


def _action_stats_from_mcts(stats: Optional[Dict]) -> List[Dict[str, Any]]:
    """
    Extract per-action statistics from the MCTS stats dict returned by search.py.

    Expected stats format (from search.py return_stats=True):
      {"top": [{"action": ("move"|"switch", obj), "n": int, "q": float, "prior": float}, ...],
       "sims": int}
    """
    if not stats:
        return []
    top = stats.get("top", []) or []
    total_n = sum(int(entry.get("n", 0)) for entry in top) or 1
    result = []
    for entry in top:
        action = entry.get("action")
        if action is None:
            continue
        kind, obj = action
        if obj is None:
            continue
        name = getattr(obj, "id", None) or getattr(obj, "species", None) or str(obj)
        n = int(entry.get("n", 0))
        result.append({
            "kind":   str(kind),
            "name":   str(name),
            "visits": n,
            "frac":   round(n / total_n, 4),
            "prior":  round(float(entry.get("prior", 0.0)), 4),
            "q":      round(float(entry.get("q", 0.0)), 4),
        })
    return result

class DataCollector:
    """
    Writes TurnRecords to a JSONL file and back-fills the outcome label
    when a battle finishes.

    Thread-safety: Not thread-safe. Use one instance per bot instance
    (each runs in its own asyncio loop / process).
    """

    def __init__(self, path: str = "data/turns.jsonl"):
        self._path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # Maps battle_id -> list of byte offsets where that battle's records start
        # (used for in-place outcome labeling — too complex; we buffer instead)
        self._pending: Dict[str, List[TurnRecord]] = {}

    def record_turn(
        self,
        *,
        battle_id: str,
        turn: int,
        eval_terms: Dict[str, float],
        eval_value: float,
        mcts_stats: Optional[Dict],
        opp_beliefs: Dict[int, Any],
        opp_active: Any,
        score_move_fn: Any,
        ctx_opp: Any,
        battle: Any,
        picked: Optional[Tuple[str, Any]],
    ) -> None:
        """
        Call this once per decision point, after MCTS returns.

        ``picked`` is the (kind, obj) tuple selected by the bot, or None.
        """
        action_stats = _action_stats_from_mcts(mcts_stats)

        # Active opponent belief entropy
        active_belief = opp_beliefs.get(id(opp_active)) if opp_beliefs and opp_active else None
        b_entropy = _belief_entropy(active_belief)

        opp_moves = _opp_top_moves(opp_beliefs, opp_active, score_move_fn, ctx_opp, battle)

        if picked is not None:
            kind, obj = picked
            name = getattr(obj, "id", None) or getattr(obj, "species", None) or str(obj)
        else:
            kind, name = "unknown", "unknown"

        record = TurnRecord(
            battle_id=battle_id,
            turn=turn,
            eval_terms=eval_terms,
            eval_value=round(eval_value, 5),
            action_stats=action_stats,
            opp_top_moves=opp_moves,
            belief_entropy=round(b_entropy, 4),
            picked_kind=str(kind),
            picked_name=str(name),
            outcome=0,
        )
        self._pending.setdefault(battle_id, []).append(record)

    def finish_battle(self, battle_id: str, won: Optional[bool]) -> None:
        """
        Stamp win/loss onto all pending records for this battle and flush to disk.

        ``won`` can be True (win), False (loss), or None (tie/unknown → 0).
        """
        records = self._pending.pop(battle_id, [])
        if not records:
            return

        if won is True:
            outcome = 1
        elif won is False:
            outcome = -1
        else:
            outcome = 0

        # Optional: discount early turns (less credit for decisions far from end)
        # For now, flat label — simplest and least biased.
        with open(self._path, "a", encoding="utf-8") as f:
            for rec in records:
                rec.outcome = outcome
                f.write(json.dumps(asdict(rec), separators=(",", ":")) + "\n")
