from __future__ import annotations

from typing import Any

from poke_env.battle import MoveCategory

from bot.model.ctx import EvalContext
from bot.scoring.damage_score import estimate_damage_fraction
from bot.scoring.helpers import remaining_count


def _move_provides_unique_value(move: Any, opp: Any, me: Any, battle: Any) -> float:
    """
    Score how uniquely valuable this move is against current opponent.
    Higher = more irreplaceable for this matchup.
    Returns value in [0.0, 1.0]
    """
    if opp is None or me is None:
        return 0.5
    
    # Get damage from this move (using real calculator)
    this_dmg = estimate_damage_fraction(move, me, opp, battle)
    
    # Compare to other available moves
    better_alternatives = 0
    similar_alternatives = 0
    
    for other_move in getattr(battle, "available_moves", []) or []:
        if other_move == move:
            continue
        
        if getattr(other_move, "category", None) == MoveCategory.STATUS:
            continue
        
        # Get damage from other move (using real calculator)
        other_dmg = estimate_damage_fraction(other_move, me, opp, battle)
        
        # If other move does >= 90% of this move's damage, it's similar
        if other_dmg >= this_dmg * 0.9:
            other_pp = getattr(other_move, "current_pp", 1)
            this_pp = getattr(move, "current_pp", 1)
            
            if other_pp > this_pp:
                better_alternatives += 1
            else:
                similar_alternatives += 1
    
    # Uniqueness score: higher when no good alternatives
    if better_alternatives > 0:
        return 0.2  # We have better options, save this PP
    elif similar_alternatives > 0:
        return 0.5  # Neutral - comparable options exist
    else:
        return 1.0  # This is our best option for this target


def pp_conservation_penalty(move: Any, battle: Any, ctx: EvalContext) -> float:
    """
    Penalty for using low-PP moves when they're not uniquely valuable.
    
    Strategy:
    - Early game (4+ Pokemon each): conserve low-PP moves
    - Late game (2- Pokemon each): spend freely
    - Mid game: context-dependent
    """
    me = ctx.me
    opp = ctx.opp
    
    if me is None or opp is None:
        return 0.0
    
    current_pp = getattr(move, "current_pp", None)
    max_pp = getattr(move, "max_pp", None)
    
    if current_pp is None or max_pp is None:
        return 0.0
    
    # If we have plenty of PP, no penalty
    if current_pp >= max_pp * 0.6:
        return 0.0
    
    # Game state
    our_remaining = remaining_count(battle.team)
    opp_remaining = remaining_count(battle.opponent_team)
    
    # Late game: no conservation needed
    if our_remaining <= 2 and opp_remaining <= 2:
        return 0.0
    
    # Early game: heavy conservation for low-PP moves
    if our_remaining >= 4 and opp_remaining >= 4:
        if max_pp <= 8:
            # This is a low-PP move (Stone Edge, Hydro Pump, Focus Blast, etc.)
            uniqueness = _move_provides_unique_value(move, opp, me, battle)
            
            # Base penalty scaled by uniqueness
            base_penalty = 20.0 * (1.0 - uniqueness)
            
            # Increase penalty if PP is very low
            if current_pp <= 2:
                base_penalty += 15.0
            elif current_pp <= 4:
                base_penalty += 8.0
            
            return base_penalty
    
    # Mid game: lighter conservation
    if max_pp <= 5:
        uniqueness = _move_provides_unique_value(move, opp, me, battle)
        penalty = 10.0 * (1.0 - uniqueness)
        
        if current_pp <= 2:
            penalty += 8.0
        
        return penalty
    
    return 0.0