from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from bot.learning.action_features import compute_action_features, ACTION_FEATURE_NAMES

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

    # Search diagnostics
    # {"root_visits_total": int, "max_visits_frac": float, "q_range": float,
    #  "n_actions": int, "n_actions_legal": int, "using_policy": bool}
    search_info: Dict[str, Any]

    # Decision
    picked_kind:      str   # "move" or "switch"
    picked_name:      str   # move id or pokemon species
    picked_is_hybrid: bool  # True if the picked action was a hybrid-branched move

    # Structured state encoding from state_encoder.encode_state_flat().
    # 312-element float list: [tokens(12×24), field(21), active_pair(3)].
    # None for records collected before the new encoder was introduced.
    state_flat: Optional[List[float]] = None

    # Outcome (filled in by finish_battle)
    # +1 = win, -1 = loss, 0 = tie / unknown
    outcome: int = 0

def _belief_entropy(belief: Any) -> float:
    """
    Normalized Shannon entropy in [0, 1] for an OpponentBelief distribution.
    0 = fully known (single candidate), 1 = highly ambiguous.
    """
    if belief is None:
        return 0.0
    dist = getattr(belief, "dist", None)
    if not dist:
        return 0.0
    ps = [max(0.0, float(p)) for _, p in dist]
    z = sum(ps)
    if z <= 1e-12:
        return 0.0
    ps = [p / z for p in ps]
    k = len(ps)
    if k <= 1:
        return 0.0
    h = 0.0
    for p in ps:
        if p > 1e-12:
            h -= p * math.log(p)
    return float(max(0.0, min(1.0, h / max(1e-12, math.log(k)))))


def _opp_top_moves(
    opp_beliefs: Dict[int, Any],
    opp_active: Any,
    k: int = 5,
) -> List[Dict[str, float]]:
    """
    Collect the opponent's predicted move distribution from the belief move pool,
    weighted by role probability.

    Returns a list of {"name": str, "prob": float} sorted descending.
    """
    if not opp_beliefs or opp_active is None:
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
    top_total = sum(p for _, p in top) or 1.0
    return [{"name": mv, "prob": round(p / top_total, 4)} for mv, p in top]


def _action_stats_from_mcts(
    stats: Optional[Dict],
    battle: Any = None,
    ctx_me: Any = None,
) -> List[Dict[str, Any]]:
    """
    Extract per-action statistics from the MCTS stats dict returned by search.py.

    Expected stats format (from search.py return_stats=True):
      {"top": [{"action": ("move"|"switch", obj), "visits": int, "q": float, "prior": float}, ...],
       "sims": int}

    battle + ctx_me: if provided, per-action features are computed and stored under "features".
    """
    if not stats:
        return []
    top = stats.get("top", []) or []
    total_n = sum(int(entry.get("visits", 0)) for entry in top) or 1
    result = []
    for entry in top:
        action = entry.get("action")
        if action is None:
            continue
        kind, obj = action[0], action[1]
        if obj is None:
            continue
        name = getattr(obj, "id", None) or getattr(obj, "species", None) or str(obj)
        n = int(entry.get("visits", 0))

        feats: Dict[str, float] = {}
        if battle is not None and ctx_me is not None:
            feats = {k: 0.0 for k in ACTION_FEATURE_NAMES}
            try:
                feats.update(compute_action_features(kind, obj, battle, ctx_me))
            except Exception:
                pass  # feats remains all-zeros with full schema
            # heuristic_score is not set by compute_action_features; fill from prior_raw
            feats["heuristic_score"] = float(entry.get("prior_raw", entry.get("prior", 0.0)))

        result.append({
            "kind":      str(kind),
            "name":      str(name),
            "visits":    n,
            "frac":      round(n / total_n, 4),
            "prior":     round(float(entry.get("prior", 0.0)), 4),
            "prior_raw": round(float(entry.get("prior_raw", entry.get("prior", 0.0))), 4),
            "q":         round(float(entry.get("q", 0.0)), 4),
            "features":  feats,
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
        opp_beliefs: Optional[Dict[int, Any]],
        opp_active: Any,
        battle: Any,
        picked: Optional[Tuple[str, Any]],
        ctx_me: Any = None,
        shadow_state: Any = None,
    ) -> None:
        """
        Call this once per decision point, after MCTS returns.

        ``picked`` is the (kind, obj) tuple selected by the bot, or None.
        ``ctx_me`` is the EvalContext for our side (used for action features).
        ``shadow_state`` is the ShadowState at this decision point; when provided,
        encode_state_flat() is called to produce the 312-dim state_flat field used
        to train the TransformerValueModel.
        """
        action_stats = _action_stats_from_mcts(mcts_stats, battle=battle, ctx_me=ctx_me)

        # Encode structured state if a ShadowState is available.
        state_flat_list: Optional[List[float]] = None
        if shadow_state is not None:
            try:
                from bot.learning.state_encoder import encode_state_flat
                flat = encode_state_flat(shadow_state)
                state_flat_list = [round(float(v), 5) for v in flat]
            except Exception:
                pass

        # Active opponent belief entropy
        active_belief = opp_beliefs.get(id(opp_active)) if opp_beliefs and opp_active else None
        b_entropy = _belief_entropy(active_belief)

        opp_moves = _opp_top_moves(opp_beliefs, opp_active)

        if picked is not None:
            kind = picked[0]
            obj  = picked[1]  # actions can be 2- or 3-tuples (kind, obj[, outcome])
            name = getattr(obj, "id", None) or getattr(obj, "species", None) or str(obj)
        else:
            kind, name = "unknown", "unknown"

        raw_search_info = (mcts_stats or {}).get("search_info", {})
        search_info: Dict[str, Any] = {
            "root_visits_total": int(raw_search_info.get("root_visits_total", 0)),
            "max_visits_frac":   round(float(raw_search_info.get("max_visits_frac", 0.0)), 4),
            "q_range":           round(float(raw_search_info.get("q_range", 0.0)), 4),
            "n_actions":         int(raw_search_info.get("n_actions", 0)),
            "n_actions_legal":   int(raw_search_info.get("n_actions_legal", 0)),
            "using_policy":      bool(raw_search_info.get("using_policy", False)),
        }

        picked_is_hybrid = bool(picked is not None and len(picked) >= 3)

        record = TurnRecord(
            battle_id=battle_id,
            turn=turn,
            eval_terms=eval_terms,
            eval_value=round(eval_value, 5),
            action_stats=action_stats,
            opp_top_moves=opp_moves,
            belief_entropy=round(b_entropy, 4),
            search_info=search_info,
            picked_kind=str(kind),
            picked_name=str(name),
            picked_is_hybrid=picked_is_hybrid,
            state_flat=state_flat_list,
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
