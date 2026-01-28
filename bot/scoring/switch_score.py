from __future__ import annotations

from typing import Any, Optional

from poke_env.battle import MoveCategory

from bot.model.ctx import EvalContext
from bot.scoring.helpers import hp_frac, remaining_count
from bot.scoring.pressure import estimate_opponent_pressure

def _estimate_damage_from_opponent(opponent: Any, target: Any, battle: Any) -> float:
    """
    Estimate the damage opponent would deal to target.
    Uses the same logic as status_score's team synergy.
    """
    if opponent is None or target is None:
        return 0.25
    
    try:
        from poke_env.calc.damage_calc_gen9 import calculate_damage
        
        # Find opponent's best move vs this target
        best_avg_damage = 0.0
        
        for move in opponent.moves.values():
            if move is None:
                continue
            
            # Skip status moves
            if getattr(move, "category", None) == MoveCategory.STATUS:
                continue
            
            try:
                # Get identifiers
                opp_id = _get_pokemon_identifier(opponent, battle)
                target_id = _get_pokemon_identifier(target, battle)
                
                if opp_id is None or target_id is None:
                    continue
                
                # Calculate damage
                min_dmg, max_dmg = calculate_damage(
                    attacker_identifier=opp_id,
                    defender_identifier=target_id,
                    move=move,
                    battle=battle,
                    is_critical=False,
                )
                
                # Convert to fraction
                target_max_hp = getattr(target, 'max_hp', None) or getattr(target, 'stats', {}).get('hp', 100)
                avg_dmg = (min_dmg + max_dmg) / 2.0
                dmg_frac = avg_dmg / target_max_hp
                
                best_avg_damage = max(best_avg_damage, dmg_frac)
                
            except Exception:
                continue
        
        if best_avg_damage > 0:
            return best_avg_damage
        else:
            # Fallback to pressure estimate
            return 0.25
    
    except Exception:
        return 0.25


def _get_pokemon_identifier(pokemon: Any, battle: Any) -> Optional[str]:
    """Get battle identifier for a Pokemon."""
    if pokemon is None or battle is None:
        return None
    
    # Check player's team
    try:
        for identifier, pkmn in battle.team.items():
            if pkmn is pokemon:
                return identifier
    except Exception:
        pass
    
    # Check opponent's team
    try:
        for identifier, pkmn in battle.opponent_team.items():
            if pkmn is pokemon:
                return identifier
    except Exception:
        pass
    
    return None

def _matchup_score(mon: Any, opponent: Any, battle: Any) -> float:
    """
    Evaluate how good mon's matchup is against opponent.
    Positive = good matchup, Negative = bad matchup
    """
    if mon is None or opponent is None:
        return 0.0
    
    score = 0.0
    
    # Defensive evaluation - how much damage does our mon take?
    damage_taken = _estimate_damage_from_opponent(opponent, mon, battle)
    
    if damage_taken < 0.15:
        score += 40  # Walls opponent completely
    elif damage_taken < 0.25:
        score += 25  # Tanks well
    elif damage_taken < 0.40:
        score += 0   # Neutral
    elif damage_taken < 0.60:
        score -= 20  # Takes heavy damage
    elif damage_taken < 0.90:
        score -= 40  # Near OHKO
    else:
        score -= 60  # OHKO'd
    
    # Offensive evaluation - how much damage can our mon deal?
    damage_dealt = _estimate_damage_from_opponent(mon, opponent, battle)
    
    if damage_dealt > 0.80:
        score += 40  # Threatens OHKO
    elif damage_dealt > 0.50:
        score += 25  # Threatens 2HKO
    elif damage_dealt > 0.30:
        score += 10  # Decent damage
    elif damage_dealt < 0.15:
        score -= 20  # Can't threaten
    
    # Speed control
    try:
        mon_speed = (mon.base_stats or {}).get("spe", 80)
        opp_speed = (opponent.base_stats or {}).get("spe", 80)
        
        if mon_speed >= opp_speed * 1.1:
            score += 15  # Outspeeds
        elif mon_speed <= opp_speed * 0.9:
            score -= 10  # Slower
    except Exception:
        pass
    
    return score

def _danger_urgency(current_mon: Any, opponent: Any, battle: Any) -> float:
    """
    How urgently do we need to switch out?
    High values = urgent switch needed
    """
    if current_mon is None or opponent is None:
        return 0.0
    
    damage = _estimate_damage_from_opponent(opponent, current_mon, battle)
    hp = hp_frac(current_mon)
    
    # Immediate danger levels
    if damage >= hp * 0.95:
        return 80  # OHKO - urgent!
    elif damage >= hp * 0.85:
        return 65  # Near OHKO
    elif damage >= hp * 0.50:
        return 40  # 2HKO
    elif damage >= hp * 0.33:
        return 20  # 3HKO
    elif damage >= hp * 0.25:
        return 10  # 4HKO
    else:
        return 0   # Tanking fine


def _switch_in_penalty(switch_target: Any, opponent: Any, battle: Any) -> float:
    """
    Penalty for taking damage on switch-in.
    High values = bad switch (takes heavy damage)
    """
    if switch_target is None or opponent is None:
        return 0.0
    
    damage = _estimate_damage_from_opponent(opponent, switch_target, battle)
    
    # Penalty scales with damage
    if damage >= 0.80:
        return 80  # Switch-in nearly dies
    elif damage >= 0.50:
        return 50  # Heavy damage
    elif damage >= 0.30:
        return 20  # Moderate damage
    elif damage >= 0.15:
        return 5   # Light chip
    else:
        return 0   # Minimal damage

def _setup_danger(mon: Any, opponent: Any, battle: Any, ctx: EvalContext) -> float:
    """
    Penalty if opponent can setup on mon.
    Positive = opponent can setup (bad for us)
    """
    if mon is None or opponent is None:
        return 0.0
    
    # Check if opponent has setup potential
    pressure = estimate_opponent_pressure(battle, ctx)
    setup_prob = pressure.setup_prob
    
    if setup_prob < 0.3:
        return 0.0  # Opponent probably doesn't have setup
    
    # Check if mon can threaten opponent (prevents setup)
    damage_dealt = _estimate_damage_from_opponent(mon, opponent, battle)
    
    if damage_dealt > 0.60:
        return -15  # We threaten opponent, hard to setup
    elif damage_dealt > 0.40:
        return 0    # Neutral
    elif damage_dealt > 0.20:
        return 15 * setup_prob  # Opponent might try to setup
    else:
        return 30 * setup_prob  # Opponent can setup freely

def _win_condition_value(switch_target: Any, battle: Any) -> float:
    """
    Bonus for preserving important Pokemon.
    Positive = preserve this mon (don't risk it)
    """
    if switch_target is None:
        return 0.0
    
    value = 0.0
    
    # High HP Pokemon are more valuable
    hp = hp_frac(switch_target)
    if hp > 0.8:
        value += 10
    
    # Don't risk your last Pokemon unnecessarily
    our_remaining = remaining_count(battle.team)
    if our_remaining <= 2:
        value += 20
    
    # TODO: Add setup sweeper detection
    # If switch_target has setup moves + good speed/attack, add value
    
    return value

def score_switch(pokemon: Any, battle: Any, ctx: EvalContext) -> float:
    """
    Score switching to a specific Pokemon.
    
    Considers:
    - Current matchup vs new matchup
    - Danger level of staying in
    - Damage taken on switch-in
    - Setup opportunities/risks
    - Win condition preservation
    
    Returns:
        Score where higher = better switch
    """
    if pokemon is None or pokemon.fainted:
        return -999.0
    
    current_mon = ctx.me
    opponent = ctx.opp
    
    if current_mon is None or opponent is None:
        # Fallback: mild preference for healthy switches
        return 10.0 * hp_frac(pokemon)
    
    # Don't switch to yourself
    if pokemon is current_mon:
        return -999.0
    
    # Matchup differential (is new matchup better?)
    current_matchup = _matchup_score(current_mon, opponent, battle)
    new_matchup = _matchup_score(pokemon, opponent, battle)
    matchup_diff = new_matchup - current_matchup
    
    # Danger urgency (do we NEED to switch?)
    danger = _danger_urgency(current_mon, opponent, battle)
    
    # Switch-in damage penalty (will switch take heavy damage?)
    switch_penalty = _switch_in_penalty(pokemon, opponent, battle)
    
    # Setup danger differential (Does the opponent setup easier against the mon we want to switch in)
    current_setup_risk = _setup_danger(current_mon, opponent, battle, ctx)
    new_setup_risk = _setup_danger(pokemon, opponent, battle, ctx)
    setup_diff = current_setup_risk - new_setup_risk
    
    # Win condition value
    preserve_value = _win_condition_value(pokemon, battle)

    score = (
        matchup_diff * 40         # Matchup improvement
        + danger * 60            # Urgency to get out
        - switch_penalty * 50    # Cost of switching in
        + setup_diff * 25        # Setup opportunity/risk
        + preserve_value * 10    # Preserve important mons
    )
    
    # ===== SPECIAL CASES =====
    
    # Regenerator bonus
    if getattr(pokemon, "ability", None) == "regenerator":
        score += 15  # Free healing
    
    # Intimidate bonus (if opponent is physical)
    if getattr(pokemon, "ability", None) == "intimidate":
        pressure = estimate_opponent_pressure(battle, ctx)
        if pressure.physical_prob > 0.5:
            score += 20 * pressure.physical_prob
    
    # Don't switch if you have a great attacking option
    # (This should be handled by comparing with move scores, but add small penalty)
    if danger < 40:  # Not in urgent danger
        score -= 15  # Slight penalty for switching when not necessary
    
    # Endgame: be more conservative with switches
    our_remaining = remaining_count(battle.team)
    opp_remaining = remaining_count(battle.opponent_team)
    if our_remaining <= 2 and opp_remaining <= 2:
        # In endgame, don't switch unless it's clearly better
        if matchup_diff < 20:
            score -= 20
    
    return score

def _slow_pivot_value(current_mon: Any, switch_target: Any, opponent: Any, battle: Any) -> float:
    """
    Calculate the value of being able to slow pivot vs hard switch.
    
    Slow pivot is valuable when:
    1. Current mon is slower than opponent
    2. Switch target would take heavy damage on hard switch
    3. Current mon can survive one hit
    4. Switch target has better matchup
    
    Returns positive value if slow pivot would be beneficial.
    """
    if current_mon is None or switch_target is None or opponent is None:
        return 0.0
    
    # Check speed relationship
    try:
        my_speed = (current_mon.base_stats or {}).get("spe", 80)
        opp_speed = (opponent.base_stats or {}).get("spe", 80)
        target_speed = (switch_target.base_stats or {}).get("spe", 80)
    except Exception:
        return 0.0
    
    # Only valuable if we're slower than opponent
    if my_speed >= opp_speed:
        return 0.0
    
    # Check if switch target would take heavy damage on hard switch
    switch_in_damage = _estimate_damage_from_opponent(opponent, switch_target, battle)
    
    if switch_in_damage < 0.3:
        return 0.0  # Switch-in doesn't care about free hit
    
    # Check if current mon can survive one hit (needed to execute slow pivot)
    my_hp = hp_frac(current_mon)
    damage_to_me = _estimate_damage_from_opponent(opponent, current_mon, battle)
    
    if damage_to_me >= my_hp * 0.95:
        return 0.0  # We die before pivoting
    
    # Check if switch target actually has a better matchup
    target_matchup = _matchup_score(switch_target, opponent, battle)
    if target_matchup < 20:
        return 0.0
    
    # Calculate slow pivot value based on damage avoided
    value = switch_in_damage * 80
    
    # Extra bonus if switch-in is fragile
    target_hp = hp_frac(switch_target)
    if target_hp < 0.6:
        value *= 1.3
    
    # Extra bonus if switch-in is faster than opponent
    if target_speed > opp_speed:
        value += 15
    
    # Scale by severity of damage avoided
    if switch_in_damage > 0.7:
        value *= 1.4  # Avoiding near-OHKO
    elif switch_in_damage > 0.5:
        value *= 1.2  # Avoiding 2HKO
    
    return value

PIVOT_MOVES = {
    "uturn", "voltswitch", "flipturn", "partingshot", 
    "batonpass", "chillyreception", "shedtail",
    "teleport",  # Gen 8+ teleport has negative priority and switches
}

def is_pivot_move(move: Any) -> bool:
    """Check if a move is a pivot move."""
    move_id = str(getattr(move, "id", "")).lower()
    return move_id in PIVOT_MOVES


def pivot_move_bonus(move: Any, battle: Any, ctx: EvalContext) -> float:
    """
    Calculate bonus for pivot moves.
    
    Pivot moves = damage + switch, so they're special.
    Slow pivot = being slower allows safe switch-in (avoids damage).
    
    This should be added to the move's damage score.
    """
    if not is_pivot_move(move):
        return 0.0
    
    me = ctx.me
    opp = ctx.opp
    
    if me is None or opp is None:
        return 0.0
    
    bonus = 0.0
    
    # 1. Momentum bonus - you maintain tempo
    bonus += 15.0
    
    # 2. Scout bonus - see opponent's response
    bonus += 10.0
    
    # 3. Evaluate best switch-in after pivot
    best_switch_score = -999.0
    best_switch_target = None
    
    for teammate in battle.team.values():
        if teammate is None or teammate.fainted or teammate is me:
            continue
        
        switch_score = score_switch(teammate, battle, ctx)
        if switch_score > best_switch_score:
            best_switch_score = switch_score
            best_switch_target = teammate
    
    # Add fraction of best switch score
    if best_switch_score > 0:
        bonus += best_switch_score * 0.3
    
    # If we're slower, pivot lets switch-in come in AFTER opponent attacks
    if best_switch_target is not None:
        slow_pivot_value = _slow_pivot_value(me, best_switch_target, opp, battle)
        bonus += slow_pivot_value
    
    # 5. Risk penalty - opponent hits you before you switch
    danger = _danger_urgency(me, opp, battle)
    if danger > 60:
        bonus -= 30  # Pivoting when in OHKO range is risky
    elif danger > 40:
        bonus -= 15  # Some risk
    
    # 6. Choice item synergy - pivot resets choice lock
    my_item = str(getattr(me, "item", "")).lower()
    if my_item in ["choiceband", "choicescarf", "choicespecs"]:
        bonus += 20  # Pivot resets choice lock!
    
    return bonus