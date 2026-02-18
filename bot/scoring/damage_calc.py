from typing import Any
import math
from poke_env.calc.damage_calc_gen9 import calculate_damage
from poke_env.battle import MoveCategory
from bot.scoring.helpers import hp_frac

def _estimate_damage_to_ally(ally: Any, opp: Any, battle: Any) -> float:
    """
    Estimate damage opponent deals to potential switch-in.
    
    Uses poke-env's built-in Gen 9 damage calculator!
    
    Returns: Average damage as fraction of ally HP (0.0 to 2.0+)
    """
    if ally is None or opp is None:
        return 0.25
    
    try:
        # Find opponent's best move vs this ally
        best_avg_damage = 0.0
        best_move = None
        
        for move in opp.moves.values():
            if move is None:
                continue
            
            # Skip status moves
            if getattr(move, "category", None) == MoveCategory.STATUS:
                continue
            
            try:
                # Use poke-env's damage calculator
                # Returns (min_damage, max_damage) as integers (HP lost)
                min_dmg, max_dmg = calculate_damage(
                    attacker_identifier=opp.species,  # or identifier
                    defender_identifier=ally.species,  # or identifier  
                    move=move,
                    battle=battle,
                    is_critical=False,
                )
                
                # Convert to fractions of max HP
                ally_max_hp = getattr(ally, 'max_hp', 100) or getattr(ally, 'stats', {}).get('hp', 100)
                
                min_frac = min_dmg / ally_max_hp
                max_frac = max_dmg / ally_max_hp
                
                # Average damage
                avg_dmg = (min_frac + max_frac) / 2.0
                
                if avg_dmg > best_avg_damage:
                    best_avg_damage = avg_dmg
                    best_move = move
                
            except Exception:
                # If calc fails for this move, continue to next
                continue
        
        if best_avg_damage > 0:
            return best_avg_damage
        else:
            # No valid moves found, fall back
            return _type_based_damage_estimate(ally, opp)
    
    except Exception:
        # Any other error, use fallback
        return _type_based_damage_estimate(ally, opp)


def _type_based_damage_estimate(ally: Any, opp: Any) -> float:
    """
    Fallback damage estimate based on type matchups.
    Used when full damage calculation fails.
    """
    try:
        from poke_env.battle import PokemonType
        from bot.scoring.helpers import safe_types
        
        opp_types = safe_types(opp)
        ally_types = safe_types(ally)
        
        # Start with average
        estimate = 0.25
        
        # Check for common walls
        if PokemonType.STEEL in ally_types:
            estimate = 0.12
            if PokemonType.FIGHTING in opp_types:
                estimate = 0.40
            elif PokemonType.FIRE in opp_types:
                estimate = 0.45
            elif PokemonType.GROUND in opp_types:
                estimate = 0.40
        
        elif PokemonType.FAIRY in ally_types:
            estimate = 0.18
            if PokemonType.DRAGON in opp_types:
                estimate = 0.05  # Immune
            elif PokemonType.STEEL in opp_types or PokemonType.POISON in opp_types:
                estimate = 0.40
        
        elif PokemonType.WATER in ally_types:
            estimate = 0.22
            if PokemonType.ELECTRIC in opp_types or PokemonType.GRASS in opp_types:
                estimate = 0.45
        
        return min(2.0, max(0.05, estimate))
    
    except Exception:
        return 0.25


def _burn_ko_threshold_value(
    ally: Any,
    opp: Any,
    battle: Any,
    damage_to_ally: float,
    phys_p: float
) -> float:
    """
    Calculate value based on how burn changes KO thresholds.
    
    Now uses ACTUAL damage from poke-env's calculator!
    
    Key insight: What matters is HITS TO KO (HTK), not raw damage.
    
    Returns:
    - Large bonus if burn saves from OHKO/2HKO (+35 to +65)
    - Small bonus if burn improves survivability meaningfully (+10 to +28)
    - Penalty if burn doesn't change HTK enough - wasted on tanky matchup (-15)
    """
    if phys_p < 0.5:
        # Not physical enough for burn to matter
        return 0.0
    
    if ally is None or opp is None:
        return 0.0
    
    ally_hp = hp_frac(ally)
    
    # Calculate hits to KO without burn
    if damage_to_ally <= 0.01:
        htk_without = 99  # Basically immune
    else:
        htk_without = math.ceil(ally_hp / damage_to_ally)
    
    # Calculate hits to KO with burn (halves physical damage)
    burned_damage = damage_to_ally * 0.5
    if burned_damage <= 0.01:
        htk_with = 99
    else:
        htk_with = math.ceil(ally_hp / burned_damage)
    
    # How many extra hits does burn give us?
    extra_hits = htk_with - htk_without
    
    # Scale by physical probability
    value = 0.0
    
    if htk_without == 1:
        # OHKO → Survives
        if htk_with >= 2:
            # Burn saves from OHKO!
            value = 50.0 * phys_p
            
            # Extra bonus if becomes 3HKO or better
            if htk_with >= 3:
                value += 15.0 * phys_p
    
    elif htk_without == 2:
        # 2HKO → More hits
        if htk_with >= 3:
            # Burn prevents 2HKO
            value = 35.0 * phys_p
            
            # Extra bonus if becomes 4HKO+
            if htk_with >= 4:
                value += 10.0 * phys_p
    
    elif htk_without == 3:
        # 3HKO → More hits
        if htk_with >= 4:
            # Meaningful improvement
            value = 20.0 * phys_p
            
            if htk_with >= 5:
                value += 8.0 * phys_p
    
    elif htk_without >= 4:
        # Already tanking well
        if extra_hits >= 2:
            # Marginal improvement
            value = 10.0 * phys_p
        else:
            # Burn barely helps - wasted on tanky matchup
            value = -15.0 * phys_p
    
    return value

def team_synergy_value_burn_section_example(ally, opp, battle, st, dist, phys_p, setup_p, priority_p):
    """
    Example of how to integrate into the burn section of team_synergy_value().
    """
    
    if st.name == "BRN":
        # Get ACTUAL damage to ally using poke-env's calculator
        damage_to_ally = _estimate_damage_to_ally(ally, opp, battle)
        
        # Calculate KO threshold value 
        ko_threshold_value = _burn_ko_threshold_value(
            ally, opp, battle, damage_to_ally, phys_p
        )
        
        # Setup deterrence
        phys_setup_p = (
            sum(w for c, w in dist if getattr(c, "is_physical", False) and getattr(c, "has_setup", False))
            if dist else 0.20
        )
        stop_sweep = 26.0 * phys_setup_p + 8.0 * setup_p * phys_p
        stop_priority = 10.0 * priority_p * phys_p
        
        benefit = ko_threshold_value + stop_sweep + stop_priority
        
        return benefit

def _calculate_ohko_probability(
    attacker: Any,
    defender: Any,
    move: Any,
    battle: Any,
    burned: bool = False
) -> float:
    """
    Calculate the exact OHKO probability using poke-env's calculator.
    
    Returns probability as float (0.0 to 1.0)
    
    Example: Your screenshot shows "37.5% chance to OHKO"
    This would return 0.375
    """
    try:
        # Calculate damage range
        min_dmg, max_dmg = calculate_damage(
            attacker_identifier=attacker.species,
            defender_identifier=defender.species,
            move=move,
            battle=battle,
            is_critical=False,
        )
        
        # Apply burn if needed
        if burned and getattr(move, 'category', None) == MoveCategory.PHYSICAL:
            min_dmg = int(min_dmg * 0.5)
            max_dmg = int(max_dmg * 0.5)
        
        # Get defender HP
        defender_max_hp = getattr(defender, 'max_hp', 100) or getattr(defender, 'stats', {}).get('hp', 100)
        defender_current_hp = int(defender_max_hp * hp_frac(defender))
        
        # Calculate OHKO probability
        # Damage rolls are from 0.85 to 1.00 (16 possible values)
        # min_dmg corresponds to 0.85 roll, max_dmg to 1.00 roll
        
        ohko_count = 0
        for i in range(16):
            # Interpolate damage for this roll
            roll_fraction = i / 15.0  # 0.0 to 1.0
            damage = int(min_dmg + (max_dmg - min_dmg) * roll_fraction)
            
            if damage >= defender_current_hp:
                ohko_count += 1
        
        return ohko_count / 16.0
    
    except Exception:
        # Fallback to simple check
        avg_dmg = (min_dmg + max_dmg) / 2.0
        return 1.0 if avg_dmg >= defender_current_hp else 0.0
