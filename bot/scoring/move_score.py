from typing import Any
import math

from poke_env.battle import MoveCategory

from bot.model.ctx import EvalContext
from bot.scoring.damage_score import (
    estimate_damage_fraction,
    ko_probability_from_fraction,
)
from bot.scoring.helpers import hp_frac, is_slower
from bot.scoring.status_score import score_status_move
from bot.scoring.secondary_score import score_secondaries
from poke_env.battle import MoveCategory, SideCondition

def score_move(move: Any, battle: Any, ctx: EvalContext) -> float:
    """
    Score a move based on its effectiveness.
    
    Status moves: Use specialized status scoring
    Damage moves: Base damage + KO bonus + secondaries + chip synergy - penalties
    """
    me = ctx.me
    opp = ctx.opp
    if me is None or opp is None:
        return -100.0

    if move.category == MoveCategory.STATUS:
        # Status value
        return score_status_move(move, battle, ctx)

    dmg_frac = float(estimate_damage_fraction(move, me, opp, battle))
    opp_hp = hp_frac(opp)

    # Base score from damage
    score = dmg_frac * 100.0
    
    # KO bonus
    ko_prob = ko_probability_from_fraction(dmg_frac, opp_hp)
    # score += ko_bonus(ko_prob, slower=is_slower(me, opp))

    if ko_prob > 0:
        # Base KO value: removing opponent is valuable
        ko_value = 50.0 * ko_prob
        
        # Speed bonus: KOing while faster means we don't take damage
        slower = is_slower(me, opp)
        speed_bonus = (5.0 if slower else 15.0) * ko_prob
        
        score += ko_value + speed_bonus

    accuracy = getattr(move, 'accuracy', 1.0) or 1.0
    
    if accuracy >= 1.0:
        score += 20  # Perfect accuracy (Earthquake, Flamethrower)
    elif accuracy >= 0.95:
        score += 15  # Near-perfect (Thunderbolt, Ice Beam)
    elif accuracy >= 0.90:
        score += 10  # Good (most standard moves)
    elif accuracy >= 0.85:
        score += 5 
    # There's no reason to score moves with an accuracy less than this, since MCTS will already branch for these (search.py)

    # Secondary effects
    if ko_prob < 0.95:
        score += score_secondaries(move, battle, ctx, ko_prob, dmg_frac=dmg_frac)
        # No need to consider secondary effects if the move is already going to KO
        # 95% chosen here because for damaging statuses like burn/toxic the residual will finish the job

    priority = getattr(move, 'priority', 0) or 0
    if priority > 0:
        # Priority moves (Aqua Jet, Mach Punch, etc.) valuable for finishing
        if opp_hp < 0.35:
            score += 20  # Excellent finisher
        elif opp_hp < 0.50:
            score += 12  # Good finisher
        else:
            score += 6   # Minor speed control value

    #Crit potential gains
    crit_bonus = calculate_crit_bonus(move, battle, ctx, dmg_frac, ko_prob)
    score += crit_bonus
    
    # Apply recoil penalty
    recoil = getattr(move, 'recoil', 0) or 0
    if recoil > 0:
        # Double-Edge (33%), Brave Bird (33%), Head Smash (50%)
        # Penalty = recoil_fraction × 50
        # 33% recoil = -16.5 points, 50% recoil = -25 points (capped at 20)
        recoil_penalty = min(20.0, abs(recoil) * 50.0)
        score -= recoil_penalty
    
    return score

def calculate_crit_bonus(move: Any, battle: Any, ctx: Any, base_damage_frac: float, ko_prob: float) -> float:
    """
    Calculate bonus value for critical hit chance.
    
    Args:
        move: The move being evaluated
        battle: Battle state
        ctx: Evaluation context
        base_damage_frac: Non-crit damage as fraction (from estimate_damage_fraction)
        ko_prob: KO probability without crit
        
    Returns:
        Bonus points (0-40)
    """
    me = ctx.me
    opp = ctx.opp
    
    if me is None or opp is None:
        return 0.0
    
    if move.category == MoveCategory.STATUS:
        return 0.0
    
    # Get crit rate
    crit_ratio = getattr(move, 'crit_ratio', 0) or 0
    crit_chance = get_crit_chance(crit_ratio)
    
    # If no meaningful crit chance, no bonus
    if crit_chance < 0.08:  # Less than high-crit moves
        return 0.0
    
    bonus = 0.0
    
    boost_value = calculate_boost_ignore_value(
        move, me, opp, battle, crit_chance
    )
    bonus += boost_value
    
    # Only consider if we don't already guaranteed KO
    if ko_prob < 0.90:
        htk_value = _calculate_htk_improvement_value(
            base_damage_frac, crit_chance, ctx
        )
        bonus += htk_value
    
    return bonus


def get_crit_chance(crit_ratio: int) -> float:
    """
    Convert crit ratio to actual probability.
    
    Gen 9 crit rates:
    - ratio 0 (normal): 1/24 = 4.17%
    - ratio 1 (Focus Energy): 1/8 = 12.5%
    - ratio 2 (Stone Edge, Razor Leaf): 1/2 = 50%
    - ratio 3+ (Frost Breath): 100%
    """
    if crit_ratio <= 0:
        return 1.0 / 24.0  # ~4.17%
    elif crit_ratio == 1:
        return 1.0 / 8.0   # 12.5%
    elif crit_ratio == 2:
        return 0.5         # 50%
    else:
        return 1.0         # 100%


def calculate_boost_ignore_value(move: Any, me: Any, opp: Any, battle: Any, crit_chance: float) -> float:
    """
    Calculate value of ignoring stat boosts/drops via crit.
    
    Crits ignore:
    - Opponent's positive defensive boosts
    - Our negative offensive boosts  
    - Screens (Reflect/Light Screen)
    
    Returns:
        Bonus points (0-25)
    """
    value = 0.0
    
    # Determine which stats matter for this move
    is_physical = move.category == MoveCategory.PHYSICAL
    
    try:
        opp_boosts = getattr(opp, 'boosts', {})
        defensive_stat = 'def' if is_physical else 'spd'
        opp_def_boost = opp_boosts.get(defensive_stat, 0)
        
        if opp_def_boost > 0:
            # Opponent is boosted defensively
            # Crit ignores defensive boosts
            
            # Value scales with boost level
            # +1 = moderate value, +2 = high value, +3+ = huge value
            boost_impact = {
                1: 8.0,   # +1 Def: crit is ~1.5x better
                2: 15.0,  # +2 Def: crit is ~2x better
                3: 22.0,  # +3 Def: crit is ~2.5x better
            }.get(min(opp_def_boost, 3), 22.0)
            
            # Scale by crit chance
            # 50% crit (Stone Edge) gets full value
            # 12% crit (Focus Energy) gets partial value
            value += boost_impact * crit_chance
    
    except Exception:
        pass
    
    try:
        my_boosts = getattr(me, 'boosts', {})
        offensive_stat = 'atk' if is_physical else 'spa'
        my_atk_boost = my_boosts.get(offensive_stat, 0)
        
        if my_atk_boost < 0:
            # We have attack drops (Intimidate, etc.)
            
            debuff_impact = {
                -1: 8.0,   # -1 Atk
                -2: 15.0,  # -2 Atk
                -3: 22.0,  # -3 Atk
            }.get(max(my_atk_boost, -3), 22.0)
            
            value += debuff_impact * crit_chance
    
    except Exception:
        pass
    
    try:
        # Check opponent's side conditions
        opp_side = battle.opponent_side_conditions if hasattr(battle, 'opponent_side_conditions') else {}
        
        # Reflect blocks physical, Light Screen blocks special
        screen_up = (
            (is_physical and SideCondition.REFLECT in opp_side) or
            (not is_physical and SideCondition.LIGHT_SCREEN in opp_side) or
            (SideCondition.AURORA_VEIL in opp_side)
        )
        
        if screen_up:
            # Screens halve damage, crits ignore them
            # This is HUGE value for high-crit moves
            value += 12.0 * crit_chance
    
    except Exception:
        pass
    
    return min(25.0, value)  # Cap at 25 points


def _calculate_htk_improvement_value(base_damage_frac: float, crit_chance: float, ctx: Any) -> float:
    """
    Calculate value of crit improving hits-to-KO.
    
    Key insight: Crits matter most when they change the plan!
    
    Examples:
    - 3HKO → 2HKO with crit: HUGE value (saves a turn)
    - 2HKO → 2HKO with crit: Small value (no plan change)
    - 5HKO → 4HKO with crit: Moderate value (marginal)
    
    Returns:
        Bonus points (0-20)
    """
    if base_damage_frac <= 0:
        return 0.0
    
    opp = ctx.opp
    opp_hp = getattr(opp, 'current_hp_fraction', 1.0) or 1.0
    
    # Calculate hits to KO without crit
    htk_no_crit = math.ceil(opp_hp / max(0.01, base_damage_frac))
    
    # Calculate hits to KO with crit (1.5x damage)
    crit_damage = base_damage_frac * 1.5
    htk_with_crit = math.ceil(opp_hp / max(0.01, crit_damage))
    
    # How many turns does crit save?
    turns_saved = htk_no_crit - htk_with_crit
    
    if turns_saved <= 0:
        # Crit doesn't change the plan
        return 0.0
    
    # Value based on what changes
    if htk_no_crit >= 4 and htk_with_crit <= 2:
        # Slow KO → Fast KO (4+ turns → 2 turns)
        # This is GAME-CHANGING
        base_value = 20.0
    
    elif htk_no_crit == 3 and htk_with_crit == 2:
        # 3HKO → 2HKO
        # Very valuable (saves a turn in race)
        base_value = 15.0
    
    elif htk_no_crit == 2 and htk_with_crit == 1:
        # 2HKO → OHKO
        # Extremely valuable (no retaliation)
        base_value = 18.0
    
    elif turns_saved == 1:
        # Any other 1-turn save
        base_value = 10.0
    
    else:
        # Saves 2+ turns
        base_value = 12.0
    
    # Scale by crit chance
    # 50% crit = expect this half the time
    # 12% crit = expect this rarely
    return base_value * crit_chance