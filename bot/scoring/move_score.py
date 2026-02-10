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
from bot.mcts.shadow_state import get_move_boosts

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
    
    setup_score = score_setup_move(move, battle, ctx)
    if setup_score > 0:
        return setup_score

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

    score -= get_stat_drop_penalty(move, battle, ctx)
    
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

def get_stat_drop_penalty(move: Any, battle: Any, ctx: EvalContext) -> float:
    """
    Penalty for moves that drop our own stats.
    
    Returns positive value to subtract from score.
    """
    
    boost_data = get_move_boosts(move)
    if not boost_data:
        return 0.0
    
    self_boosts, target_boosts, chance = boost_data
    if not self_boosts:
        return 0.0
    
    # Calculate penalty
    penalty = 0.0
    for stat, stages in self_boosts.items():
        if stages < 0:  # Only penalize drops
            if stat in ['atk', 'spa']:
                penalty += abs(stages) * 15.0  # -2 SpA = -30 points
            elif stat == 'spe':
                penalty += abs(stages) * 10.0
            else:
                penalty += abs(stages) * 5.0
    
    # Reduce penalty if opponent almost dead
    opp_hp = getattr(ctx.opp, 'current_hp_fraction', 1.0)
    if opp_hp < 0.3:
        penalty *= 0.5  # Worth it for the KO
    
    # Increase penalty in sweep scenarios
    my_hp = getattr(ctx.me, 'current_hp_fraction', 1.0)
    if my_hp > 0.7 and opp_hp > 0.5:
        penalty *= 1.3  # Need sustained damage
    
    return penalty

def score_setup_move(move: Any, battle: Any, ctx: EvalContext) -> float:
    """
    Score stat-boosting moves (Swords Dance, Nasty Plot, Dragon Dance, etc.)
    
    Value depends on:
    - How many stages we boost
    - Which stats we boost (offensive > defensive)
    - Whether we can survive to use the boosts (risk assessment)
    - Current HP situation
    NOTE: will probably add scoring for how moves improve current matchup too
    """
    
    boost_data = get_move_boosts(move)
    if not boost_data:
        return 0.0
    
    self_boosts, target_boosts, chance = boost_data
    
    # We only care about self-boosts for setup moves
    if not self_boosts:
        return 0.0
    
    # Only positive boosts (setup moves, not Draco Meteor penalties)
    if all(v <= 0 for v in self_boosts.values()):
        return 0.0
    
    me = ctx.me
    opp = ctx.opp
    
    current_boosts = getattr(me, 'boosts', {})
    
    # Base value of boosts WITH DIMINISHING RETURNS
    boost_value = 0.0
    
    for stat, stages in self_boosts.items():
        if stages > 0:
            current_level = current_boosts.get(stat, 0)
            
            # Don't boost if already at +6
            if current_level >= 6:
                continue
            
            # Clamp to +6 max
            actual_stages = min(stages, 6 - current_level)
            
            # Calculate value (the benefits of boosting when already boosted diminish with the more boosts)
            base_per_stage = 30.0 if stat in ['atk', 'spa'] else 25.0 if stat == 'spe' else 20.0
            
            for i in range(actual_stages):
                new_level = current_level + i + 1
                
                if new_level <= 2:
                    multiplier = 1.0
                elif new_level == 3:
                    multiplier = 0.7
                elif new_level == 4:
                    multiplier = 0.5
                elif new_level == 5:
                    multiplier = 0.3
                else:  # new_level >= 6
                    multiplier = 0.1
                
                boost_value += base_per_stage * multiplier
    
    # If we got no value (already at +6), return 0
    if boost_value <= 0:
        return 0.0
    
    # Risk factor: Can we survive to use the boosts?
    my_hp = getattr(me, 'current_hp_fraction', 1.0)
    opp_hp = getattr(opp, 'current_hp_fraction', 1.0)
    
    # Estimate opponent's best damage against us
    opp_max_damage = 0.0
    for opp_move in getattr(opp, 'moves', {}).values():
        try:
            dmg = estimate_damage_fraction(opp_move, opp, me, battle)
            opp_max_damage = max(opp_max_damage, dmg)
        except:
            opp_max_damage = 0.5
    
    # Apply risk penalty
    if opp_max_damage >= my_hp:
        boost_value *= 0.2
    elif opp_max_damage >= my_hp * 0.75:
        boost_value *= 0.4
    elif opp_max_damage >= my_hp * 0.5:
        boost_value *= 0.6
    else:
        boost_value *= 1.2
    
    # Speed tier consideration
    try:
        my_spe = me.stats.get('spe', 100)
        opp_spe = opp.stats.get('spe', 100)
        
        if my_spe > opp_spe:
            boost_value *= 1.2
        else:
            boost_value *= 0.9
    except:
        pass
    
    # Multi-stat boost bonus
    num_boosted_stats = sum(1 for v in self_boosts.values() if v > 0)
    if num_boosted_stats >= 2:
        boost_value *= 1.3
    
    # HP situation
    if my_hp > 0.8 and opp_hp > 0.6:
        boost_value *= 1.2
    elif opp_hp < 0.3:
        boost_value *= 0.3
    
    return boost_value

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