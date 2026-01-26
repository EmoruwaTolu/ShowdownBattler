from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poke_env.battle import MoveCategory

from bot.model.ctx import EvalContext
from bot.scoring.damage_score import estimate_damage_fraction
from bot.scoring.helpers import hp_frac

from bot.scoring.pressure import estimate_opponent_pressure


@dataclass(frozen=True)
class DamageRace:
    tko_opp: float          # expected turns for us to KO opponent (continuous)
    ttd_me: float           # expected turns for opponent to KO us (continuous)
    state: str              # "WINNING" | "LOSING" | "CLOSE"
    move_exp_dmg: float     # expected damage fraction (acc-weighted) for this move
    move_priority: int      # this move's priority


def _base_speed(mon: Any) -> float:
    try:
        return float((getattr(mon, "base_stats", None) or {}).get("spe", 80))
    except Exception:
        return 80.0


def _speed_order(me: Any, opp: Any) -> int:
    """
    +1 if we likely act first (faster), -1 if they likely act first, 0 if tie/unknown.
    Deadzone avoids flip-flopping near ties.
    """
    ms, os = _base_speed(me), _base_speed(opp)
    if ms >= os * 1.05:
        return +1
    if os >= ms * 1.05:
        return -1
    return 0


def _initiative_penalty(
    *,
    tko_opp: float,
    ttd_me: float,
    effective_order: int,
    opp_priority_p: float,
) -> float:
    """
    Convert "moving second is bad" into a stable penalty.

    Key idea:
      If both sides KO in ~the same number of turns (e.g., 2HKO vs 2HKO),
      the side that acts first typically wins the tie.

    We therefore add an extra penalty when:
      - we're likely acting second (effective_order == -1)
      - the race is close (tko_opp ~ ttd_me)
      - opponent might have priority (so even if we're faster, ties are less safe)
    """
    # how close is the race?
    diff = abs(tko_opp - ttd_me)

    # closeness factor: 1.0 when very close, fades to 0.0 when not close
    # (tuned for "2HKO vs 2HKO" type ranges)
    close01 = max(0.0, min(1.0, 1.0 - (diff / 1.2)))

    # baseline tie-break penalty for moving second
    if effective_order == -1:
        # stronger when close
        pen = 0.90 * close01
    elif effective_order == +1:
        # small bonus (implemented as negative penalty)
        pen = -0.35 * close01
    else:
        pen = 0.0

    # if opponent likely has priority, our "going first" is less reliable
    # (this pushes CLOSE situations slightly toward LOSING)
    pen += 0.45 * opp_priority_p * close01

    return pen


def evaluate_race_for_move(battle: Any, ctx: EvalContext, move: Any) -> DamageRace:
    """
    Priority-aware, set-aware "race" evaluation for THIS move.

    Goals:
      - Treat "2HKO vs 2HKO but they're faster" as losing more reliably.
      - Use set-aware pressure (setup/priority) for opponent damage-to-us estimate.
    """
    me, opp = ctx.me, ctx.opp
    if me is None or opp is None:
        return DamageRace(99.0, 99.0, "CLOSE", 0.0, 0)

    # Status moves don't have a damage race; treat as "CLOSE"
    if getattr(move, "category", None) == MoveCategory.STATUS:
        return DamageRace(99.0, 99.0, "CLOSE", 0.0, int(getattr(move, "priority", 0) or 0))

    my_hp = max(0.01, hp_frac(me))
    opp_hp = max(0.01, hp_frac(opp))

    # Our move expected damage (using real calculator)
    dmg = float(estimate_damage_fraction(move, me, opp, battle))
    acc = float(getattr(move, "accuracy", 1.0) or 1.0)
    acc = max(0.0, min(1.0, acc))
    exp_dmg = dmg * acc

    tko_opp = 99.0 if exp_dmg <= 1e-9 else (opp_hp / exp_dmg)

    # Opponent pressure: now set-aware (includes setup/priority likelihood)
    pressure = estimate_opponent_pressure(battle, ctx)
    opp_pressure = float(getattr(pressure, "damage_to_me_frac", 0.26))
    opp_priority_p = float(getattr(pressure, "priority_prob", 0.15))

    ttd_me = 99.0 if opp_pressure <= 1e-9 else (my_hp / opp_pressure)

    prio = int(getattr(move, "priority", 0) or 0)

    # Baseline speed order
    order = _speed_order(me, opp)

    # Our move having priority means we act first *this turn* (ignoring opponent priority)
    effective_order = order
    if prio > 0:
        effective_order = +1

    # Light "turn order" shaping (kept from your version)
    if effective_order == -1:
        tko_opp += 0.55
    elif effective_order == +1:
        ttd_me += 0.55

    # Stronger "tie-break" shaping for close races + opponent priority risk
    tko_opp += _initiative_penalty(
        tko_opp=tko_opp,
        ttd_me=ttd_me,
        effective_order=effective_order,
        opp_priority_p=opp_priority_p,
    )

    # 1HKO-ish corrections
    one_hit_them = (exp_dmg + 1e-6) >= opp_hp
    one_hit_us = (opp_pressure + 1e-6) >= my_hp

    if one_hit_them and effective_order == -1:
        # if we need to land a KO but likely move second, it's less reliable
        tko_opp += 1.10

    if one_hit_us and effective_order == +1:
        # if we likely move first, we slightly "buy" a turn vs getting deleted
        ttd_me += 0.90

    # Decide state
    margin = 0.6
    if tko_opp + margin < ttd_me:
        state = "WINNING"
    elif ttd_me + margin < tko_opp:
        state = "LOSING"
    else:
        state = "CLOSE"

    return DamageRace(
        tko_opp=float(tko_opp),
        ttd_me=float(ttd_me),
        state=state,
        move_exp_dmg=float(exp_dmg),
        move_priority=int(prio),
    )