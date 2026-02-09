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

def calculate_miss_cost(status_value: float, accuracy: float, me: Any, opp: Any) -> float:
    """
    Calculate cost of missing a status move.
    
    Miss cost increases when:
    - Status is more valuable (higher opportunity cost)
    - We're damaged (less time to waste turns)
    - We're slower (opponent gets free hit)
    - Move has lower accuracy (worse when you miss)
    
    Returns:
        Penalty points (typically 15-50)
    """
    # Base cost
    cost = 15.0
    
    # Scale with status value
    # High-value status (55 pts) → higher miss cost
    # Low-value status (20 pts) → lower miss cost
    cost += status_value * 0.2
    
    # Scale with miss chance
    # 50% acc (50% miss) → 1.5x cost
    miss_chance = 1.0 - accuracy
    cost *= (1.0 + miss_chance * 0.5)
    
    # Penalty when damaged (wasting turns is worse) NOTE: will probs upgrade this to look and see if the opp is threatening KO instead
    my_hp = hp_frac(me)
    if my_hp < 0.7:
        cost += (1.0 - my_hp) * 15.0
    
    # Penalty when slower (opponent gets free damage)
    if is_slower(me, opp):
        cost += 8.0
    
    return cost

def score_status_move(move: Any, battle: Any, ctx: EvalContext) -> float:
    """
    Main status move scoring function.
    
    Components:
    1. Base status value (includes near/long-term via consolidated functions)
    2. Team synergy
    3. Tempo risk
    4. Race state modifier (scales with degree of winning/losing)
    5. Miss cost (scales with status value and accuracy)
    """
    opp = ctx.opp
    me = ctx.me
    if opp is None:
        return -100.0

    # Check what status this move inflicts
    status = getattr(move, 'status', None)

    if getattr(opp, 'status', None) is not None:
        return -120.0
    
    if not status_is_applicable(status, move, opp):
        return -80.0

    score = get_base_status_value(status, me, opp, ctx)

    accuracy = getattr(move, 'accuracy', 1.0) or 1.0
    
    if accuracy < 1.0:
        # Miss cost scales with how valuable the status is
        # Missing a powerful status move is worse than missing a weak one
        miss_cost = calculate_miss_cost(score, accuracy, me, opp)
        
        # Expected value formula: EV = accuracy × value + (1-accuracy) × (-miss_cost)
        score = accuracy * score + (1.0 - accuracy) * (-miss_cost)
    
    return score

def burn_immediate_value(me, opp):
    """Calculate burn value from actual moves (or base stats fallback)."""
    value = 20.0
    
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


def _estimate_damage_to_ally(ally: Any, opp: Any, battle: Any) -> float:
    """
    Estimate damage opponent deals to potential switch-in.
    
    If poke-env 0.10.0+ is available: Uses built-in damage calculator
    Otherwise: Falls back to pressure estimation
    
    Returns: Fraction of ally HP opponent deals per turn (0.0 to 2.0+)
    """
    if ally is None or opp is None:
        return 0.25
    
    # Try real damage calculator if available (poke-env 0.10.0+)
    if HAS_DAMAGE_CALC:
        try:
            # Get identifiers for both Pokemon (e.g., "p1: Suicune", "p2: Salamence")
            ally_identifier = _get_pokemon_identifier(ally, battle)
            opp_identifier = _get_pokemon_identifier(opp, battle)
            
            if ally_identifier is None or opp_identifier is None:
                # Couldn't get identifiers, fall back
                return _estimate_damage_via_pressure(ally, opp, battle)
            
            # Find opponent's best move vs this ally
            best_avg_damage = 0.0
            
            for move in opp.moves.values():
                if move is None:
                    continue
                
                # Skip status moves
                if getattr(move, "category", None) == MoveCategory.STATUS:
                    continue
                
                try:
                    # Returns (min_damage, max_damage) as integers (HP lost)
                    min_dmg, max_dmg = calculate_damage(
                        attacker_identifier=opp_identifier,
                        defender_identifier=ally_identifier,
                        move=move,
                        battle=battle,
                        is_critical=False,
                    )
                    
                    # Convert to fractions of max HP
                    ally_max_hp = getattr(ally, 'max_hp', None)
                    if ally_max_hp is None or ally_max_hp <= 0:
                        ally_max_hp = getattr(ally, 'stats', {}).get('hp', 100)
                    
                    if ally_max_hp <= 0:
                        continue
                    
                    min_frac = min_dmg / ally_max_hp
                    max_frac = max_dmg / ally_max_hp
                    
                    # Average damage
                    avg_dmg = (min_frac + max_frac) / 2.0
                    
                    if avg_dmg > best_avg_damage:
                        best_avg_damage = avg_dmg
                    
                except Exception:
                    # If calc fails for this move, continue to next
                    continue
            
            if best_avg_damage > 0:
                return best_avg_damage
            else:
                # No valid moves found, fall back
                return _type_based_damage_estimate(ally, opp)
        
        except Exception:
            # Any other error with damage calc, fall back to pressure
            return _estimate_damage_via_pressure(ally, opp, battle)
    
    else:
        # No damage calculator available, use pressure estimation
        return _estimate_damage_via_pressure(ally, opp, battle)


def _estimate_damage_via_pressure(ally: Any, opp: Any, battle: Any) -> float:
    """
    Estimate damage using pressure calculation (fallback for poke-env < 0.10.0).
    
    Returns: Fraction of ally HP opponent deals per turn (0.0 to 2.0+)
    """
    try:
        # Create temporary context with ally as active Pokemon
        temp_ctx = EvalContext(
            me=ally,
            opp=opp,
            battle=battle, 
            cache={},  # Fresh cache for this calculation
        )
        
        # Run full pressure estimation
        pressure = estimate_opponent_pressure(battle, temp_ctx)
        
        # Return the damage fraction
        return pressure.damage_to_me_frac
    
    except Exception:
        # Ultimate fallback to type-based heuristic
        return _type_based_damage_estimate(ally, opp)


def _type_based_damage_estimate(ally: Any, opp: Any) -> float:
    """
    Fallback damage estimate based on type matchups.
    Used when full pressure calculation fails.
    """
    try:
        opp_types = safe_types(opp)
        ally_types = safe_types(ally)
        
        # Start with average
        estimate = 0.25
        
        # Check for common walls
        if PokemonType.STEEL in ally_types:
            # Steel resists many types
            estimate = 0.12
            
            # But weak to Fighting/Fire/Ground
            if PokemonType.FIGHTING in opp_types:
                estimate = 0.40
            elif PokemonType.FIRE in opp_types:
                estimate = 0.45
            elif PokemonType.GROUND in opp_types:
                estimate = 0.40
        
        elif PokemonType.FAIRY in ally_types:
            # Fairy resists Fighting/Bug/Dark, immune to Dragon
            estimate = 0.18
            
            if PokemonType.DRAGON in opp_types:
                estimate = 0.05  # Immune to Dragon
            elif PokemonType.STEEL in opp_types or PokemonType.POISON in opp_types:
                estimate = 0.40  # Weak to Steel/Poison
        
        elif PokemonType.WATER in ally_types:
            estimate = 0.22
            
            if PokemonType.ELECTRIC in opp_types or PokemonType.GRASS in opp_types:
                estimate = 0.45
        
        # More type checks could be added here
        
        return min(2.0, max(0.05, estimate))
    
    except Exception:
        # Ultimate fallback
        return 0.25