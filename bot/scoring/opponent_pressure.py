from dataclasses import dataclass
from typing import Any

from bot.scoring.helpers import hp_frac, physical_probability
from bot.model.opponent_model import get_opponent_set_distribution


@dataclass(frozen=True)
class OpponentPressure:
    """
    Summary of how threatening the opponent's current active is
    if we do NOT immediately neutralize them.
    """
    dpt: float          # estimated damage per turn (fraction of HP)
    lethal_in: float    # estimated turns until we faint
    setup_p: float      # probability opponent set has setup
    priority_p: float   # probability opponent set has priority

    @property
    def is_immediate_threat(self) -> bool:
        return self.lethal_in <= 2.0

    @property
    def is_setup_threat(self) -> bool:
        return self.setup_p >= 0.35

    @property
    def is_priority_threat(self) -> bool:
        return self.priority_p >= 0.30


def estimate_opponent_pressure(
    battle: Any,
    ctx: Any,
    default_gen: int = 9,
) -> OpponentPressure:
    me = ctx.me
    opp = ctx.opp

    if me is None or opp is None:
        return OpponentPressure(
            dpt=0.20,
            lethal_in=5.0,
            setup_p=0.0,
            priority_p=0.0,
        )

    gen = getattr(getattr(battle, "format", None), "gen", None)
    if gen is None:
        gen = getattr(getattr(ctx, "battle", None), "gen", default_gen) or default_gen

    try:
        dist = get_opponent_set_distribution(opp, int(gen)) or []
    except Exception:
        dist = []

    setup_p = sum(w for c, w in dist if getattr(c, "has_setup", False))
    priority_p = sum(w for c, w in dist if getattr(c, "has_priority", False))

    phys_p = physical_probability(opp, battle, ctx)

    dpt = 0.24
    dpt += 0.07 * phys_p
    dpt += 0.03 * (1.0 - phys_p)
    dpt += 0.05 * setup_p

    try:
        boosts = getattr(opp, "boosts", None) or {}
        if any(v > 0 for v in boosts.values()):
            dpt += 0.06
    except Exception:
        pass

    dpt = max(0.08, min(0.65, dpt))

    my_hp = max(0.01, hp_frac(me))
    lethal_in = my_hp / max(1e-6, dpt)

    return OpponentPressure(
        dpt=dpt,
        lethal_in=lethal_in,
        setup_p=setup_p,
        priority_p=priority_p,
    )
