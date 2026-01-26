from typing import Any
from poke_env.calc.damage_calc_gen9 import calculate_damage
from poke_env.battle import MoveCategory


def _get_pokemon_identifier(pokemon: Any, battle: Any) -> str:
    """
    Get the battle identifier for a Pokemon (e.g., "p1: Gengar", "p2: Zacian").
    
    Returns:
        Identifier string, or None if not found
    """
    if pokemon is None or battle is None:
        return None
    
    # Check player's team
    try:
        for identifier, pkmn in battle.team.items():
            if pkmn is pokemon:
                return identifier
    except Exception as e:
        print(f"Error checking player team: {e}")
        pass
    
    # Check opponent's team
    try:
        for identifier, pkmn in battle.opponent_team.items():
            if pkmn is pokemon:
                return identifier
    except Exception as e:
        print(f"Error checking opponent team: {e}")
        pass
    
    # DEBUG: If not found, print why
    print(f"⚠️ Pokemon not found: {getattr(pokemon, 'species', 'unknown')}")
    print(f"   Player team: {[f'{k}: {v.species}' for k, v in battle.team.items()]}")
    print(f"   Opponent team: {[f'{k}: {v.species}' for k, v in battle.opponent_team.items()]}")
    print(f"   Looking for id: {id(pokemon)}")
    for k, v in battle.team.items():
        print(f"   {k} id: {id(v)}, match: {v is pokemon}")
    for k, v in battle.opponent_team.items():
        print(f"   {k} id: {id(v)}, match: {v is pokemon}")
    
    return None


def estimate_damage_fraction(move: Any, me: Any, opp: Any, battle: Any) -> float:
    """
    Calculate damage fraction using poke-env's Gen 9 damage calculator.
    
    The calculator already handles:
    - Multi-hit moves (returns total damage)
    - Type effectiveness
    - STAB
    - Abilities
    - Items
    - Weather/Terrain
    - Screens
    - Everything else!
    
    Args:
        move: The move being used
        me: The attacking Pokemon
        opp: The defending Pokemon
        battle: The battle object (required for damage calculator)
        
    Returns:
        Estimated fraction of opponent HP removed (0.0 to 2.0+)
    """
    # Status moves deal no damage
    if move.category == MoveCategory.STATUS:
        return 0.0
    
    # Moves with 0 base power deal no damage
    bp = float(getattr(move, "base_power", 0) or 0.0)
    if bp <= 0:
        return 0.0
    
    # Get Pokemon identifiers for damage calculator
    me_identifier = _get_pokemon_identifier(me, battle)
    opp_identifier = _get_pokemon_identifier(opp, battle)
    
    if me_identifier is None or opp_identifier is None:
        # Couldn't find Pokemon in battle - return conservative estimate
        return 0.25
    
    try:
        # Use poke-env's damage calculator
        # This already handles multi-hit moves and returns total damage!
        min_dmg, max_dmg = calculate_damage(
            attacker_identifier=me_identifier,
            defender_identifier=opp_identifier,
            move=move,
            battle=battle,
            is_critical=False,
        )
        
        # Get opponent's max HP
        opp_max_hp = getattr(opp, 'max_hp', None)
        if opp_max_hp is None or opp_max_hp <= 0:
            opp_max_hp = getattr(opp, 'stats', {}).get('hp', 100)
        
        if opp_max_hp <= 0:
            return 0.25
        
        # Return average damage as fraction of HP
        avg_dmg = (min_dmg + max_dmg) / 2.0
        return avg_dmg / opp_max_hp
        
    except Exception as e:
        # If damage calculator fails, return conservative estimate
        # This should rarely happen in real battles
        return 0.25


def ko_probability_from_fraction(dmg_frac: float, opp_hp_frac: float) -> float:
    """
    Calculate KO probability from damage fraction and opponent HP.
    
    Since the calculator gives us average damage, we approximate the roll range
    as 85%-100% of the calculated average (standard Pokemon damage rolls).
    
    Args:
        dmg_frac: Damage as fraction of max HP (from estimate_damage_fraction)
        opp_hp_frac: Opponent's current HP as fraction of max HP
        
    Returns:
        Probability of KO (0.0 to 1.0)
    """
    if opp_hp_frac <= 0:
        return 1.0
    
    # The damage calculator returns average damage
    # Actual damage rolls from 85% to 100% of this average
    # (This is the standard Pokemon damage roll range)
    min_frac = dmg_frac * 0.85
    max_frac = dmg_frac * 1.00
    
    # If minimum damage KOs, guaranteed KO
    if min_frac >= opp_hp_frac:
        return 1.0
    
    # If maximum damage can't KO, no chance
    if max_frac < opp_hp_frac:
        return 0.0
    
    # Linear interpolation for partial KO chance
    # How much of the roll range results in a KO?
    range_size = max_frac - min_frac
    if range_size <= 0:
        return 0.0
    
    ko_range = max_frac - opp_hp_frac
    ko_prob = ko_range / range_size
    
    return max(0.0, min(1.0, ko_prob))


def ko_bonus(ko_prob: float, slower: bool) -> float:
    """
    Calculate bonus points for KO probability.
    
    KOs are valuable because they:
    - Remove an opponent's Pokemon
    - Give us a free switch
    - Reduce opponent's options
    
    Args:
        ko_prob: Probability of KO (0.0 to 1.0)
        slower: True if we're slower than opponent
        
    Returns:
        Bonus score points
    """
    if ko_prob <= 0:
        return 0.0
    
    # Base KO value
    base = 50.0 * ko_prob
    
    # Speed bonus: KOing while faster is better (we get the KO before taking damage)
    # If slower, still good but less valuable
    speed = (5.0 if slower else 20.0) * ko_prob
    
    return base + speed