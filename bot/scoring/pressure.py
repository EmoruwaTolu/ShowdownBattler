from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bot.model.ctx import EvalContext
from bot.model.opponent_model import get_opponent_set_distribution
from bot.scoring.helpers import hp_frac, physical_probability


@dataclass(frozen=True)
class OpponentPressure:
    """
    A compact summary of "how scary is the opponent's current active for us right now".

    damage_to_me_frac: estimated fraction of our HP we lose per opponent attacking turn.
    physical_prob:     P(opponent is primarily physical | candidate sets)
    setup_prob:        P(opponent has a setup move | candidate sets)
    priority_prob:     P(opponent has priority | candidate sets)
    threat:            normalized 0..1 convenience scalar (used for "pressure" knobs)
    """
    damage_to_me_frac: float
    physical_prob: float
    setup_prob: float
    priority_prob: float
    threat: float


def estimate_opponent_pressure(battle: Any, ctx: EvalContext, default_gen: int = 9) -> OpponentPressure:
    opp = ctx.opp
    me = ctx.me

    if opp is None or me is None:
        return OpponentPressure(0.25, 0.5, 0.2, 0.1, 0.5)

    phys_p = float(physical_probability(opp, battle, ctx))

    gen = getattr(getattr(battle, "format", None), "gen", None)
    if gen is None:
        gen = getattr(getattr(ctx, "battle", None), "gen", default_gen) or default_gen

    try:
        dist = get_opponent_set_distribution(opp, int(gen)) or []
    except Exception:
        dist = []

    setup_p = float(sum(w for c, w in dist if getattr(c, "has_setup", False))) if dist else 0.25
    prio_p = float(sum(w for c, w in dist if getattr(c, "has_priority", False))) if dist else 0.15

    # Base "damage per opponent turn" guess (relative urgency knob).
    base = 0.24
    base += 0.10 * setup_p      # DD/CM/etc.
    base += 0.06 * prio_p       # priority trading risk
    base += 0.05 * phys_p       # burn-relevant pressure

    # Any boosts => more pressure (covers special boosts too)
    try:
        boosts = getattr(opp, "boosts", None) or {}
        if any(v > 0 for v in boosts.values()):
            base += 0.06
    except Exception:
        pass

    base += (1.0 - hp_frac(me)) * 0.06

    dmg = max(0.08, min(0.65, base))
    threat = (dmg - 0.10) / 0.55
    threat = max(0.0, min(1.0, threat))

    return OpponentPressure(
        damage_to_me_frac=float(dmg),
        physical_prob=float(max(0.0, min(1.0, phys_p))),
        setup_prob=float(max(0.0, min(1.0, setup_p))),
        priority_prob=float(max(0.0, min(1.0, prio_p))),
        threat=float(threat),
    )
