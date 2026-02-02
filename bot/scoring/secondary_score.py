from __future__ import annotations

from typing import Any, Dict, List, Optional

from poke_env.battle import Status

from bot.model.ctx import EvalContext
from bot.scoring.damage_score import estimate_damage_fraction
from bot.scoring.helpers import hp_frac
from bot.scoring.status_score import major_status_is_applicable, team_synergy_value, _burn_chip_damage_value
from bot.scoring.race import evaluate_race_for_move
from bot.scoring.pressure import estimate_opponent_pressure
from bot.scoring.chip_score import chip_synergy_value

# Race state multipliers (reduced to avoid double-counting with TTD swing)
RACE_LOSING_MULTIPLIER = 1.05   # Reduced from 1.12
RACE_WINNING_MULTIPLIER = 0.95  # Reduced from 0.93
RACE_CLOSE_MULTIPLIER = 1.00

# Team synergy weights (how much secondary synergy matters vs primary)
SYNERGY_WEIGHT_BURN_PARA = 0.70
SYNERGY_WEIGHT_SLEEP_FREEZE = 0.30
SYNERGY_WEIGHT_POISON = 0.40

# Chip synergy weights (unlock revenge KOs after damage + status)
CHIP_WEIGHT_BURN_PARA = 0.35
CHIP_WEIGHT_SLEEP_FREEZE = 0.20
CHIP_WEIGHT_POISON = 0.25

def _parse_secondary_status(sec: Dict[str, Any]) -> Optional[Status]:
    s = sec.get("status", None)
    if not s:
        return None
    try:
        return Status[s.upper()]
    except Exception:
        return None


def _secondary_chance(sec: Dict[str, Any]) -> float:
    if "chance" not in sec:
        return 1.0
    try:
        return max(0.0, min(1.0, float(sec["chance"]) / 100.0))
    except Exception:
        return 0.0


def _ttd_from_frac(my_hp_frac: float, opp_dmg_frac: float) -> float:
    if opp_dmg_frac <= 1e-9:
        return 99.0
    return my_hp_frac / opp_dmg_frac


def _burn_ttd_swing_value(battle: Any, ctx: EvalContext, pressure) -> float:
    me, opp = ctx.me, ctx.opp
    if opp is None or me is None:
        return 0.0

    phys_p = pressure.physical_prob
    my_hp = hp_frac(me)
    opp_dpt = pressure.damage_to_me_frac

    ttd_before = _ttd_from_frac(my_hp, opp_dpt)

    if opp_dpt >= my_hp:
        return 0.0

    remaining_after_first = my_hp - opp_dpt
    opp_dpt_after = opp_dpt * (0.5 * phys_p + 1.0 * (1.0 - phys_p))
    ttd_after = 1.0 + _ttd_from_frac(remaining_after_first, opp_dpt_after)

    delta = max(0.0, ttd_after - ttd_before)
    value = delta * 12.0
    if int(ttd_after) > int(ttd_before):
        value += 18.0

    return value


def _paralysis_speed_swing_value(battle: Any, ctx: EvalContext, pressure) -> float:
    """
    Para pressure is driven by setup_prob + threat (NOT phys_p),
    because special attackers can also setup.
    """
    me, opp = ctx.me, ctx.opp
    if opp is None or me is None:
        return 0.0

    try:
        my_spe = (me.base_stats or {}).get("spe", 80)
        opp_spe = (opp.base_stats or {}).get("spe", 80)
    except Exception:
        my_spe, opp_spe = 80, 80

    was_slower = my_spe < opp_spe
    becomes_faster = my_spe >= (opp_spe * 0.5)  # Para HALVES speed in Gen 7+

    value = 0.0
    if becomes_faster and was_slower:
        value += 22.0
    elif becomes_faster:
        value += 6.0

    setup_p = float(getattr(pressure, "setup_prob", 0.0))
    threat = float(getattr(pressure, "threat", 0.5))
    pressure01 = max(0.0, min(1.0, 0.65 * setup_p + 0.35 * threat))

    value += 0.25 * (10.0 + 12.0 * pressure01)
    return value


def _secondary_base_value(status: Status, battle: Any, ctx: EvalContext, pressure) -> float:
    """Base value for secondary status effects."""
    if status == Status.BRN:
        phys_p = pressure.physical_prob
        base = 10.0 + 10.0 * phys_p

        # Only add TTD swing value if opponent is physical
        if phys_p >= 0.35:
            base += _burn_ttd_swing_value(battle, ctx, pressure)
        
        # Add burn chip damage value
        chip_value = _burn_chip_damage_value(battle, ctx)
        base += chip_value * 0.6
        
        return base

    if status == Status.PAR:
        base = 12.0
        base += _paralysis_speed_swing_value(battle, ctx, pressure)
        return base

    if status in (Status.PSN, Status.TOX):
        return 9.0
    if status == Status.SLP:
        return 18.0
    if status == Status.FRZ:
        return 14.0

    return 0.0


def _team_synergy_bonus_for(status: Status, battle: Any, ctx: EvalContext) -> float:
    class _DummyMove:
        def __init__(self, st: Status):
            self.status = st

    return float(team_synergy_value(_DummyMove(status), battle, ctx))


def score_secondaries(
    move: Any,
    battle: Any,
    ctx: EvalContext,
    ko_prob: float,
    dmg_frac: Optional[float] = None,
) -> float:
    """
    Calculate expected value of secondary status effects.
    
    Formula for each secondary:
        EV = chance * (1 - ko_prob) * accuracy * status_value
    
    Example: Scald (80 BP Water move with 30% burn)
        damage_ev = dmg_frac * 100.0
        burn_value = base + synergy + chip
        burn_ev = 0.30 * (1 - ko_prob) * 1.0 * burn_value
        total = damage_ev + burn_ev
    
    Args:
        move: The move being evaluated
        battle: Current battle state
        ctx: Evaluation context
        ko_prob: Probability this move KOs the opponent
        dmg_frac: Damage fraction (computed if not provided)
    
    Returns:
        Total expected value from all secondary effects
    """
    opp = ctx.opp
    me = ctx.me
    if opp is None or me is None:
        return 0.0

    secs: List[Dict[str, Any]] = getattr(move, "secondary", None) or []
    if not secs:
        return 0.0

    if dmg_frac is None:
        dmg_frac = float(estimate_damage_fraction(move, me, opp, battle))
    dmg_frac = float(max(0.0, dmg_frac))

    no_ko_prob = max(0.0, 1.0 - float(ko_prob))

    acc = float(getattr(move, "accuracy", 1.0) or 1.0)
    acc = max(0.0, min(1.0, acc))

    # Get or cache pressure (expensive to compute)
    cache_key = "opponent_pressure"
    if cache_key in ctx.cache:
        pressure = ctx.cache[cache_key]
    else:
        pressure = estimate_opponent_pressure(battle, ctx)
        ctx.cache[cache_key] = pressure

    try:
        race_state = evaluate_race_for_move(battle, ctx, move).state
    except Exception:
        race_state = "CLOSE"

    if race_state == "LOSING":
        race_mult = RACE_LOSING_MULTIPLIER
    elif race_state == "WINNING":
        race_mult = RACE_WINNING_MULTIPLIER
    else:
        race_mult = RACE_CLOSE_MULTIPLIER

    total = 0.0
    for sec in secs:
        status = _parse_secondary_status(sec)
        if status is None:
            continue
        if not major_status_is_applicable(status, move, opp):
            continue

        chance = _secondary_chance(sec)
        if chance <= 0:
            continue

        eff = chance * no_ko_prob * acc

        base = _secondary_base_value(status, battle, ctx, pressure)
        synergy = _team_synergy_bonus_for(status, battle, ctx)

        chip = chip_synergy_value(
            battle=battle,
            ctx=ctx,
            damage_dealt_frac=dmg_frac,
            after_status=status,
        )

        if status in (Status.BRN, Status.PAR):
            synergy_w = SYNERGY_WEIGHT_BURN_PARA
            chip_w = CHIP_WEIGHT_BURN_PARA
        elif status in (Status.SLP, Status.FRZ):
            synergy_w = SYNERGY_WEIGHT_SLEEP_FREEZE
            chip_w = CHIP_WEIGHT_SLEEP_FREEZE
        else:
            synergy_w = SYNERGY_WEIGHT_POISON
            chip_w = CHIP_WEIGHT_POISON

        total += eff * race_mult * (base + synergy_w * synergy + chip_w * chip)

    return total