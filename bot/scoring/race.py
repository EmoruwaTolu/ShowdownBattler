from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poke_env.battle import MoveCategory

from bot.model.ctx import EvalContext
from bot.scoring.damage_score import estimate_damage_fraction
from bot.scoring.helpers import hp_frac



@dataclass(frozen=True)
class DamageRace:
    tko_opp: float          # expected turns for us to KO opponent (continuous)
    ttd_me: float           # expected turns for opponent to KO us (continuous)
    state: str              # "WINNING" | "LOSING" | "CLOSE"
    move_exp_dmg: float     # expected damage fraction (acc-weighted) for this move
    move_priority: int      # this move's priority


@dataclass(frozen=True)
class OppThreat:
    best_exp_dmg_frac: float   # accuracy-weighted damage fraction into `target`
    priority_prob: float       # rough probability opponent uses priority this turn
    best_is_priority: bool     # whether best-damage option is priority
    best_priority: int         # highest priority among threatening moves


def estimate_opp_threat_into_target(
    battle: Any,
    *,
    opp: Any,
    target: Any,
) -> OppThreat:
    """
    Estimate opponent's threat into a *specific target* using the same Gen9
    damage calculator path as `estimate_damage_fraction`.

    We take the maximum accuracy-weighted damage fraction over the opponent's
    damaging moves.

    This is suitable for:
      - race evaluation (ttd_me for current mon or a prospective switch-in)
      - basic priority risk shaping
    """
    if opp is None or target is None:
        return OppThreat(0.25, 0.0, False, 0)

    target_hp = max(0.01, float(hp_frac(target)))

    best = 0.0
    best_is_prio = False

    prio_best = 0.0
    prio_level = 0

    for mv in (getattr(opp, "moves", None) or {}).values():
        if mv is None:
            continue
        if getattr(mv, "category", None) == MoveCategory.STATUS:
            continue

        dmg = float(estimate_damage_fraction(mv, opp, target, battle))
        acc = float(getattr(mv, "accuracy", 1.0) or 1.0)
        acc = max(0.0, min(1.0, acc))
        exp = dmg * acc

        if exp > best:
            best = exp
            best_is_prio = (int(getattr(mv, "priority", 0) or 0) > 0)

        pr = int(getattr(mv, "priority", 0) or 0)
        if pr > 0:
            prio_best = max(prio_best, exp)
            prio_level = max(prio_level, pr)

    # Priority probability proxy:
    priority_prob = 0.0
    if prio_best > 1e-9:
        if prio_best >= target_hp * 0.95:
            priority_prob = 0.80
        elif prio_best >= target_hp * 0.60:
            priority_prob = 0.45
        else:
            priority_prob = 0.15

    # Fallback to conservative DPT if no damaging moves were usable
    if best <= 1e-9:
        best = 0.25

    return OppThreat(
        best_exp_dmg_frac=float(best),
        priority_prob=float(priority_prob),
        best_is_priority=bool(best_is_prio),
        best_priority=int(prio_level),
    )


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



def evaluate_race_for_move(
    battle: Any,
    ctx: EvalContext,
    move: Any,
    *,
    me_override: Any | None = None,
    opp_override: Any | None = None,
) -> DamageRace:
    """
    Priority-aware "race" evaluation for THIS move.

    Changes:
      A) `ttd_me` is computed from opponent's best expected damage INTO the
         specific target (current mon or a prospective switch-in).
      B) Priority risk is derived from opponent's actual priority move threat
         into that target (not a generic pressure prior).

    Pass `me_override` to evaluate a prospective switch-in vs the current opponent.
    """
    me = me_override if me_override is not None else ctx.me
    opp = opp_override if opp_override is not None else ctx.opp

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

    # A) Opponent threat into THIS target
    threat = estimate_opp_threat_into_target(battle, opp=opp, target=me)
    opp_pressure = float(threat.best_exp_dmg_frac)
    opp_priority_p = float(threat.priority_prob)

    ttd_me = 99.0 if opp_pressure <= 1e-9 else (my_hp / opp_pressure)

    prio = int(getattr(move, "priority", 0) or 0)

    # Baseline speed order
    order = _speed_order(me, opp)

    # Our move having priority means we act first *this turn* (ignoring opponent priority)
    effective_order = order
    if prio > 0:
        effective_order = +1

    # B) If opponent likely has priority, our "going first" is less reliable.
    if effective_order == +1 and opp_priority_p >= 0.45:
        effective_order = 0

    # Light turn-order shaping
    if effective_order == -1:
        tko_opp += 0.55
    elif effective_order == +1:
        ttd_me += 0.55

    # Tie-break shaping for close races + opponent priority risk
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
        tko_opp += 1.10

    if one_hit_us and effective_order == +1:
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
