from typing import Any, Optional
import math

from poke_env.battle import MoveCategory
from poke_env.battle import Status
from poke_env.battle import PokemonType

try:
    from poke_env.calc.damage_calc_gen9 import calculate_damage
    HAS_DAMAGE_CALC = True
except ImportError:
    HAS_DAMAGE_CALC = False
    print("WARNING: poke-env damage calculator not available (requires 0.10.0+)")
    print("Falling back to pressure-based damage estimation")

from bot.model.ctx import EvalContext
from bot.scoring.helpers import (
    hp_frac,
    safe_types,
    is_slower,
)
from bot.scoring.pressure import estimate_opponent_pressure

try:
    from poke_env.data import GenData
    _TYPE_CHART = GenData.from_gen(9).type_chart
except Exception:
    _TYPE_CHART = {}

def status_is_applicable(status: Status, move: Any, opp: Any) -> bool:
    """
    Check if status can be applied (type immunities).
    
    Returns False if:
    - Fire-type vs Burn
    - Electric-type vs Paralysis
    - Steel/Poison-type vs Poison
    - Ground-type vs Thunder Wave (if not Mold Breaker)
    """
    opp_types = safe_types(opp)
    
    if status == Status.BRN:
        return PokemonType.FIRE not in opp_types
    
    elif status == Status.PAR:
        if PokemonType.ELECTRIC in opp_types:
            return False
        
        # Thunder Wave specifically blocked by Ground
        if getattr(move, 'type', None) == PokemonType.ELECTRIC:
            if PokemonType.GROUND in opp_types:
                # Check for Mold Breaker/immunity-ignoring effect
                ignore_immunity = getattr(move, 'ignore_immunity', False)
                return ignore_immunity
        
        return True
    
    elif status in (Status.PSN, Status.TOX):
        return (PokemonType.STEEL not in opp_types and 
                PokemonType.POISON not in opp_types)
    
    # Sleep, Freeze have no type immunities
    return True

def get_base_status_value(status: Status, me: Any, opp: Any, ctx: EvalContext) -> float:
    """
    Get immediate tactical value of inflicting status.
    """
    if status == Status.BRN:
        return burn_immediate_value(me, opp)
    
    elif status == Status.PAR:
        return para_immediate_value(me, opp)
    
    elif status in (Status.PSN, Status.TOX):
        return poison_immediate_value(status)
    
    elif status == Status.SLP:
        return sleep_immediate_value()
    
    elif status == Status.FRZ:
        return freeze_immediate_value()
    
    return 20.0  # Default status value

def calculate_miss_cost(
    status_value: float,
    accuracy: float,
    me: Any,
    opp: Any,
    pressure: Optional[Any] = None,
) -> float:
    """
    Calculate cost of missing a status move.

    Miss cost increases when:
    - Status is more valuable (higher opportunity cost)
    - Opponent threatens a KO / near-KO (wasting a turn is fatal)
    - We're slower (opponent gets free damage)
    - Move has lower accuracy (bad when you miss)

    Args:
        pressure: OpponentPressure from estimate_opponent_pressure(); when supplied,
                  replaces the old HP-only heuristic with threat-aware scaling.

    Returns:
        Penalty points (typically 15-70)
    """
    cost = 15.0

    # Scale with status value
    cost += status_value * 0.2

    # Scale with miss chance (50% acc → ×1.5 cost)
    miss_chance = 1.0 - accuracy
    cost *= (1.0 + miss_chance * 0.5)

    # Threat-based urgency: how many turns can we survive?
    # Using pressure.damage_to_me_frac (per-turn HP fraction we lose).
    my_hp = hp_frac(me)
    if pressure is not None:
        dmg = max(1e-6, float(pressure.damage_to_me_frac))
        turns_to_ko = my_hp / dmg
        if turns_to_ko <= 1.2:
            # Very likely dying this turn / next — clicking status is almost certainly wrong
            cost += 35.0
        elif turns_to_ko <= 2.5:
            # 2HKO range: still very dangerous to waste a turn
            cost += 18.0
        elif turns_to_ko <= 4.0:
            # Moderate pressure: some urgency
            cost += 6.0
        # If turns_to_ko > 4: passive opponent, no extra penalty
    else:
        # Legacy fallback: HP-only
        if my_hp < 0.7:
            cost += (1.0 - my_hp) * 15.0

    # Penalty when slower (opponent gets a free hit on the miss turn)
    if is_slower(me, opp):
        cost += 8.0

    return cost

def _absorber_multiplier(status: Status, move: Any, battle: Any, opp: Any, me: Any = None) -> float:
    """
    Returns a <1.0 multiplier when the opponent has an alive bench mon that is
    type-immune to the status and can safely switch in to absorb it.

    The multiplier scales with the absorber's *effective* HP:
      - Effective HP = actual HP × 0.5 if our active threatens it SE (unsafe pivot)
      - High effective HP (>=50%): 0.70 — safe, healthy absorber
      - Mid  effective HP (25–50%): 0.82 — real but risky pivot
      - Low  effective HP (<25%):   0.92 — barely alive or can't safely switch in
    """
    opp_team = getattr(battle, "opponent_team", {}) or {}
    me_types = safe_types(me) if me is not None else set()

    best_effective_hp = 0.0
    for bench_mon in opp_team.values():
        if bench_mon is opp:
            continue
        bench_hp = float(getattr(bench_mon, "current_hp_fraction", 0) or 0)
        if bench_hp <= 0:
            continue
        if not status_is_applicable(status, move, bench_mon):
            # Check if our active can threaten this absorber SE.
            # If yes, the absorber can't safely pivot in — halve its effective weight.
            effective_hp = bench_hp
            if _TYPE_CHART and me_types:
                bench_types = safe_types(bench_mon)
                if bench_types:
                    try:
                        for mt in me_types:
                            mult = 1.0
                            for bt in bench_types:
                                mult *= float(PokemonType.damage_multiplier(
                                    mt, bt, type_chart=_TYPE_CHART))
                            if mult >= 2.0:
                                effective_hp *= 0.5  # SE threat: risky switch-in
                                break
                    except Exception:
                        pass

            if effective_hp > best_effective_hp:
                best_effective_hp = effective_hp

    if best_effective_hp <= 0:
        return 1.0
    if best_effective_hp >= 0.50:
        return 0.70   # Safe, healthy absorber: very likely to pivot in
    if best_effective_hp >= 0.25:
        return 0.82   # Real but risky pivot option
    return 0.92       # Barely viable; risky switch-in for the opponent


def score_status_move(move: Any, battle: Any, ctx: EvalContext) -> float:
    """
    Main status move scoring function.

    Components:
    1. Base status value (type-adjusted for burn/para)
    2. Pressure gate: drops score sharply when opponent threatens KO
    3. Absorber penalty: reduces score if opponent bench can absorb status
    4. Miss cost: threat-aware EV via calculate_miss_cost
    """
    opp = ctx.opp
    me = ctx.me
    if opp is None:
        return -100.0

    status = getattr(move, 'status', None)

    if getattr(opp, 'status', None) is not None:
        return -120.0

    if not status_is_applicable(status, move, opp):
        return -80.0

    score = get_base_status_value(status, me, opp, ctx)

    # Pressure gate: how many turns until the opponent KOs us? If we're about to be deleted, spending a turn on status is wrong.
    pressure = None
    try:
        pressure = estimate_opponent_pressure(battle, ctx)
    except Exception:
        pass

    turns_to_ko = float("inf")
    if pressure is not None:
        my_hp = hp_frac(me)
        dmg = max(1e-6, float(pressure.damage_to_me_frac))
        turns_to_ko = my_hp / dmg
        if turns_to_ko <= 1.2:
            # Opponent likely KOs us this very turn — status is almost certainly wrong
            score *= 0.15
        elif turns_to_ko <= 2.5:
            # 2HKO: highly risky to give up a turn
            score *= 0.50
        elif turns_to_ko <= 4.0:
            # Moderate pressure: modest discount
            score *= 0.80
        # > 4 turns: passive opponent, no penalty

    # PAR clutch bump: paralysis can flip turn order and is worth more under pressure than the gate implies. 
    # If we're slower, landing PAR removes the opponent's biggest advantage — partially recover the score the gate discounted.
    if status == Status.PAR and is_slower(me, opp):
        if turns_to_ko <= 2.5:
            score *= 1.30   # Heavy pressure but PAR could save us
        elif turns_to_ko <= 4.0:
            score *= 1.15   # Moderate pressure, speed flip still very valuable

    # Absorber penalty: reduce score if opponent bench can absorb status
    # Floor at 0.65: even with a healthy absorber, status still has merit
    score *= max(0.65, _absorber_multiplier(status, move, battle, opp, me=me))

    # Expected value with threat-aware miss cost
    accuracy = getattr(move, 'accuracy', 1.0) or 1.0

    if accuracy < 1.0:
        miss_cost = calculate_miss_cost(score, accuracy, me, opp, pressure=pressure)
        score = accuracy * score + (1.0 - accuracy) * (-miss_cost)

    return score

def burn_immediate_value(me, opp):
    """
    Calculate burn value from actual moves (or base stats fallback).

    Extra value sources:
    - Opponent revealed physical moves (move-list analysis)
    - Opponent base stats skewed toward Atk (early-game fallback)
    - Opponent has positive Atk boosts (Swords Dance etc.) — burn cancels them
    """
    value = 20.0

    # Boost check: if opponent already has Atk stages, burn is worth even more
    # (halving an already-boosted attack is a massive tempo swing)
    try:
        boosts = getattr(opp, "boosts", {}) or {}
        atk_boost = int(boosts.get("atk", 0))
        if atk_boost > 0:
            value += min(15.0, 5.0 * atk_boost)
    except Exception:
        pass

    opp_moves = getattr(opp, 'moves', {})
    if not opp_moves:
        # Fallback to base stats
        opp_atk = (opp.base_stats or {}).get("atk", 100)
        opp_spa = (opp.base_stats or {}).get("spa", 100)
        if opp_atk > opp_spa * 1.15:
            value += 25.0
        return value

    # Count physical power
    total_power = 0
    physical_power = 0

    for move in opp_moves.values():
        power = getattr(move, 'base_power', 0) or 0
        if power == 0:
            continue

        total_power += power
        if getattr(move, 'category', None) == MoveCategory.PHYSICAL:
            physical_power += power

    # Scale bonus with physical percentage
    if total_power > 0:
        physical_pct = physical_power / total_power
        value += 40.0 * physical_pct

    return value

def para_immediate_value(me: Any, opp: Any) -> float:
    """
    Immediate value of paralyzing opponent.
    
    Components:
    1. Speed control (25% full para chance per turn)
    2. Speed halving (if we're slower, now we're faster!)
    """
    value = 20.0  # Base value (25% full para chance)
    
    # Check if we're slower
    try:
        if is_slower(me, opp):
            value += 20.0
    except:
        # Can't determine speed, moderate value
        value += 10.0
    
    return value

def poison_immediate_value(status: Status) -> float:
    """
    Immediate value of poisoning opponent.
    
    Regular Poison: 1/8 HP per turn
    Toxic: Ramping (1/16, 2/16, 3/16, ...)
    """
    if status == Status.TOX:
        # Toxic ramps up - more valuable
        return 35.0
    else:
        # Regular poison
        return 28.0


def sleep_immediate_value() -> float:
    """
    Immediate value of putting opponent to sleep.
    
    Sleep: Opponent can't move for 1-3 turns (huge!)
    """
    return 55.0  # Very high value - opponent loses turns

def freeze_immediate_value() -> float:
    """
    Immediate value of freezing opponent.
    
    Freeze: Similar to sleep (20% thaw chance per turn)
    """
    return 45.0  # High value but slightly less than sleep

def _get_pokemon_identifier(pokemon: Any, battle: Any) -> Optional[str]:
    """
    Get the identifier for a Pokemon object.
    
    Identifiers have format like "p1: Suicune" or "p2: Salamence".
    This is needed for poke-env's damage calculator.
    
    :param pokemon: Pokemon object to get identifier for
    :param battle: Battle object containing team dictionaries
    :return: Identifier string or None if not found
    """
    if pokemon is None:
        return None
    
    # Check player's team
    for identifier, pkmn in battle.team.items():
        if pkmn is pokemon:
            return identifier
    
    # Check opponent's team
    for identifier, pkmn in battle.opponent_team.items():
        if pkmn is pokemon:
            return identifier
    
    return None