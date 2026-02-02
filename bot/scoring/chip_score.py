from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from poke_env.battle import MoveCategory
from poke_env.battle import Status

from bot.model.ctx import EvalContext
from bot.scoring.damage_score import estimate_damage_fraction, ko_probability_from_fraction
from bot.scoring.helpers import hp_frac
from bot.scoring.pressure import estimate_opponent_pressure


@dataclass(frozen=True)
class RevengeOption:
    ko_prob: float
    is_priority: bool
    is_faster: bool
    exp_damage: float


def _base_speed(p: Any, default: int = 80) -> float:
    try:
        return float((p.base_stats or {}).get("spe", default))
    except Exception:
        return float(default)


def _best_revenge_option(
    *,
    battle: Any,
    ctx: EvalContext,
    opp_hp: float,
    after_status: Optional[Status],
) -> RevengeOption:
    """
    Best immediate KO line from our team given opponent HP.
    If after_status == PAR, we treat opponent as half-speed for speed checks.
    """
    opp = ctx.opp
    if opp is None:
        return RevengeOption(0.0, False, False, 0.0)

    opp_spe = _base_speed(opp)
    if after_status == Status.PAR:
        opp_spe *= 0.5  # Para HALVES speed in Gen 7+

    best = RevengeOption(0.0, False, False, 0.0)

    for ally in getattr(battle, "team", {}).values():
        if ally is None:
            continue
        try:
            if ally.fainted:
                continue
        except Exception:
            pass

        ally_spe = _base_speed(ally)
        moves = getattr(ally, "moves", None) or {}

        for mv in moves.values():
            if mv is None:
                continue
            if getattr(mv, "category", None) == MoveCategory.STATUS:
                continue

            # Use real damage calculator
            exp = float(estimate_damage_fraction(mv, ally, opp, battle))
            if exp <= 0.0:
                continue

            acc = float(getattr(mv, "accuracy", 1.0) or 1.0)
            acc = max(0.0, min(1.0, acc))
            exp_eff = exp * acc

            ko_prob = ko_probability_from_fraction(exp_eff, opp_hp)
            prio = int(getattr(mv, "priority", 0) or 0) > 0
            faster = ally_spe >= opp_spe

            # Must have either speed OR priority to count as revenge.
            if not prio and not faster:
                continue

            score = (
                ko_prob * 100.0
                + (15.0 if faster else 0.0)
                + (12.0 if prio else 0.0)
                + exp_eff * 20.0
            )

            best_score = (
                best.ko_prob * 100.0
                + (15.0 if best.is_faster else 0.0)
                + (12.0 if best.is_priority else 0.0)
                + best.exp_damage * 20.0
            )

            if score > best_score:
                best = RevengeOption(float(ko_prob), prio, faster, float(exp_eff))

    return best


def chip_synergy_value(
    *,
    battle: Any,
    ctx: EvalContext,
    damage_dealt_frac: float,
    after_status: Optional[Status] = None,
) -> float:
    """
    Reward chip because it unlocks clean revenge/cleanup lines.

    Value is based on:
      best_revenge_KO_prob(after chip) - best_revenge_KO_prob(before chip)

    PLUS explicit support for "my follow-up priority" lines (Scald -> Aqua Jet),
    gated by whether we likely survive one hit to click that priority move.
    """
    opp = ctx.opp
    me = ctx.me
    if opp is None or me is None:
        return 0.0

    opp_hp_now = hp_frac(opp)
    dmg = max(0.0, float(damage_dealt_frac))
    opp_hp_after = max(0.0, opp_hp_now - dmg)

    before = _best_revenge_option(battle=battle, ctx=ctx, opp_hp=opp_hp_now, after_status=after_status)
    after = _best_revenge_option(battle=battle, ctx=ctx, opp_hp=opp_hp_after, after_status=after_status)

    delta = max(0.0, after.ko_prob - before.ko_prob)

    value = 38.0 * delta

    if before.ko_prob < 0.25 and after.ko_prob >= 0.90:
        value += 18.0
    elif before.ko_prob < 0.50 and after.ko_prob >= 0.75:
        value += 10.0

    if after.is_priority and after.ko_prob > before.ko_prob:
        value += 6.0
    if after.is_faster and after.ko_prob > before.ko_prob:
        value += 4.0

    # Self follow-up priority (Scald -> Aqua Jet)
    pressure = estimate_opponent_pressure(battle, ctx)
    my_hp = hp_frac(me)

    # rough "can we eat one hit and then click priority?"
    can_follow_up = (my_hp - pressure.damage_to_me_frac) > 0.05

    if can_follow_up:
        best_prio_before = 0.0
        best_prio_after = 0.0

        for mv in getattr(battle, "available_moves", []) or []:
            if mv is None:
                continue
            if getattr(mv, "category", None) == MoveCategory.STATUS:
                continue
            if int(getattr(mv, "priority", 0) or 0) <= 0:
                continue

            # Use real damage calculator
            exp = float(estimate_damage_fraction(mv, me, opp, battle))
            if exp <= 0.0:
                continue

            best_prio_before = max(best_prio_before, ko_probability_from_fraction(exp, opp_hp_now))
            best_prio_after = max(best_prio_after, ko_probability_from_fraction(exp, opp_hp_after))

        pr_delta = max(0.0, best_prio_after - best_prio_before)
        value += 30.0 * pr_delta

        if best_prio_before < 0.25 and best_prio_after >= 0.90:
            value += 12.0
        
        # Role Preservation Bonus
        # If chip + priority keeps us alive vs dying to status, huge bonus!
        # This captures: "Scald -> Aqua Jet keeps Suicune alive" vs "Thunder Wave -> die"
        if best_prio_after > 0.5:
            # We have a good priority follow-up KO line
            # This keeps us in the game vs clicking status and dying
            value += 25.0

    return value