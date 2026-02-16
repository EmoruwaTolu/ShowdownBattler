import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple
from functools import lru_cache

# MoveCategory mock for standalone testing
class MoveCategory:
    PHYSICAL = "PHYSICAL"
    SPECIAL = "SPECIAL"
    STATUS = "STATUS"

def load_randbats_database(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the gen9randombattle.json database.
    
    Args:
        path: Optional path to JSON file. If None, searches common locations.
    
    Returns:
        Dictionary mapping species name -> randbats data
    """
    if path is None:
        # Search common locations
        candidates = [
            os.getenv("RANDBATS_DB_PATH", ""),
            "gen9randombattle.json",
            "/mnt/user-data/uploads/gen9randombattle.json",
            "bot/data/gen9randombattle.json",
        ]
    else:
        candidates = [path]
    
    for candidate in candidates:
        if not candidate:
            continue
        try:
            if os.path.exists(candidate):
                with open(candidate, 'r') as f:
                    data = json.load(f)
                # print(f"✓ Loaded randbats database: {len(data)} Pokemon")
                return data
        except Exception as e:
            print(f"Failed to load {candidate}: {e}")
    
    print("⚠ Warning: Could not load randbats database, using fallback")
    return {}


# Global database - loaded once
RANDBATS_DB = load_randbats_database()

# These are incomplete - the real solution is to check poke_env's move database
# But for common moves, we can hardcode them
KNOWN_PHYSICAL_MOVES = {
    'iceshard', 'knockoff', 'lowkick', 'tripleaxel', 'earthquake', 'stoneedge',
    'closecombat', 'extremespeed', 'meteormash', 'uturn', 'gunkshot', 'liquidation',
    'aquajet', 'machpunch', 'bulletpunch', 'suckerpunch', 'shadowsneak', 
    'quickattack', 'firstimpression', 'drainpunch', 'facade', 'stompingtantrum',
    'rocktomb', 'rockslide', 'rockblast', 'powerwhip', 'seedbomb', 'woodhammer',
    'ironhead', 'heavyslam', 'playrough', 'bodyslam', 'doubleedge', 'return',
    'gigaimpact', 'flareblitz', 'firefang', 'firepunch', 'thunderpunch', 'icepunch',
    'psychocut', 'zenheadbutt', 'avalanche', 'iciclespear', 'waterfallclimb',
    'crunch', 'darkestlariat', 'payback', 'throatchop', 'xscissor', 'leechlife',
    'poisonjab', 'drillrun', 'dragonclaw', 'outrage', 'dragontail',
}

KNOWN_SPECIAL_MOVES = {
    'fireblast', 'icebeam', 'thunderbolt', 'hydropump', 'surf', 'scald',
    'sludgebomb', 'earthpower', 'flashcannon', 'darkpulse', 'dragonpulse',
    'moonblast', 'dazzlinggleam', 'psychic', 'psyshock', 'focusblast',
    'energyball', 'gigadrain', 'leafstorm', 'hurricane', 'airslash',
    'shadowball', 'hex', 'thunderwave', 'discharge', 'voltswitch',
    'flamethrower', 'lavaplume', 'overheat', 'blizzard', 'freezedry',
    'waterpulse', 'muddywater', 'weatherball', 'aurasphere', 'vacuumwave',
    'bugbuzz', 'signalbeam', 'ancientpower', 'powergem', 'chargebeam',
    'dracometeor', 'spacialrend', 'judgment', 'technoblast',
}

SETUP_MOVES = {
    'swordsdance', 'nastyplot', 'dragondance', 'calmmind', 'bulkup', 'quiverdance',
    'shellsmash', 'bellydrum', 'shiftgear', 'agility', 'tailglow', 'coil', 'curse', 'growth',
}

PRIORITY_MOVES = {
    'extremespeed', 'aquajet', 'machpunch', 'iceshard', 'suckerpunch', 'bulletpunch',
    'shadowsneak', 'quickattack', 'vacuumwave', 'firstimpression',
}

HAZARD_MOVES = {
    'stealthrock', 'spikes', 'toxicspikes', 'stickyweb',
}

REMOVAL_MOVES = {
    'rapidspin', 'defog',
}

def normalize_move_name(move_name: str) -> str:
    """Normalize move name to lowercase, no spaces/dashes."""
    return move_name.lower().replace(' ', '').replace('-', '')

@lru_cache(maxsize=512)
def is_physical_move_cached(move_name: str) -> bool:
    """Check if a move is physical (cached)."""
    normalized = normalize_move_name(move_name)
    return normalized in KNOWN_PHYSICAL_MOVES

@lru_cache(maxsize=512)
def is_special_move_cached(move_name: str) -> bool:
    """Check if a move is special (cached)."""
    normalized = normalize_move_name(move_name)
    return normalized in KNOWN_SPECIAL_MOVES

def get_species_data(mon: Any) -> Optional[Dict[str, Any]]:
    """
    Get randbats data for a Pokemon.
    
    Args:
        mon: Pokemon object with .species attribute
    
    Returns:
        Randbats data dict or None if not found
    """
    if not RANDBATS_DB:
        return None
    
    species = getattr(mon, 'species', None)
    if not species:
        return None
    
    # Handle forme names (e.g., "Landorus-Therian" -> "Landorus-Therian")
    return RANDBATS_DB.get(species)

def get_all_possible_moves(mon: Any) -> Set[str]:
    """
    Get all possible moves this Pokemon could have across all roles.
    
    Args:
        mon: Pokemon object
    
    Returns:
        Set of normalized move names
    """
    data = get_species_data(mon)
    if not data:
        return set()
    
    all_moves = set()
    for role_data in data.get('roles', {}).values():
        for move in role_data.get('moves', []):
            all_moves.add(normalize_move_name(move))
    
    return all_moves

def get_role_names(mon: Any) -> List[str]:
    """
    Get list of role names for this Pokemon.
    
    Args:
        mon: Pokemon object
    
    Returns:
        List of role names (e.g., ["Fast Attacker", "Setup Sweeper"])
    """
    data = get_species_data(mon)
    if not data:
        return []
    
    return list(data.get('roles', {}).keys())

def is_physical_attacker(mon: Any) -> bool:
    """
    Check if Pokemon is primarily a physical attacker based on possible moves.
    
    Args:
        mon: Pokemon object
    
    Returns:
        True if physical attacker, False otherwise
    """
    # First try using base stats (fast fallback)
    try:
        atk = mon.base_stats.get('atk', 100)
        spa = mon.base_stats.get('spa', 100)
        if atk > spa * 1.2:  # 20% higher Attack
            return True
        if spa > atk * 1.2:  # 20% higher SpA
            return False
    except:
        pass
    
    # Use movepool analysis
    all_moves = get_all_possible_moves(mon)
    if not all_moves:
        # Fallback: check known moves
        physical_count = 0
        special_count = 0
        for move in (getattr(mon, 'moves', None) or {}).values():
            if getattr(move, 'category', None) == MoveCategory.PHYSICAL:
                physical_count += 1
            elif getattr(move, 'category', None) == MoveCategory.SPECIAL:
                special_count += 1
        return physical_count > special_count
    
    physical_count = sum(1 for m in all_moves if is_physical_move_cached(m))
    special_count = sum(1 for m in all_moves if is_special_move_cached(m))
    
    # If we can't determine, check role names
    if physical_count == special_count:
        roles = get_role_names(mon)
        if any('Physical' in r for r in roles):
            return True
        if any('Special' in r for r in roles):
            return False
        # Default to base stats comparison
        try:
            atk = mon.base_stats.get('atk', 100)
            spa = mon.base_stats.get('spa', 100)
            return atk >= spa
        except:
            pass
    
    return physical_count > special_count

def is_fast_sweeper(mon: Any) -> bool:
    """
    Check if Pokemon is a fast sweeper based on role and base speed.
    
    Args:
        mon: Pokemon object
    
    Returns:
        True if fast sweeper, False otherwise
    """
    # Check role names first (most reliable)
    roles = get_role_names(mon)
    if any('Fast' in r for r in roles):
        return True
    if any('Setup Sweeper' in r for r in roles):
        # Setup sweepers often become fast after boosting
        return True
    
    # Fallback to base speed
    try:
        spe = mon.base_stats.get('spe', 100)
        return spe >= 100
    except:
        return False

def is_defensive(mon: Any) -> bool:
    """
    Check if Pokemon is defensive/bulky based on role and stats.
    
    Args:
        mon: Pokemon object
    
    Returns:
        True if defensive, False otherwise
    """
    # Check role names
    roles = get_role_names(mon)
    if any('Bulky' in r or 'Support' in r or 'Wall' in r for r in roles):
        return True
    
    # Fallback to bulk calculation
    try:
        hp = mon.base_stats.get('hp', 100)
        defense = mon.base_stats.get('def', 100)
        spdef = mon.base_stats.get('spd', 100)
        bulk = hp * (defense + spdef) / 2
        return bulk > 15000
    except:
        return False

def has_setup_potential(mon: Any) -> bool:
    """
    Check if Pokemon can use setup moves.
    
    Args:
        mon: Pokemon object
    
    Returns:
        True if has setup moves in movepool
    """
    all_moves = get_all_possible_moves(mon)
    return bool(all_moves & SETUP_MOVES)

def has_priority_moves(mon: Any) -> bool:
    """
    Check if Pokemon has priority moves.
    
    Args:
        mon: Pokemon object
    
    Returns:
        True if has priority moves in movepool
    """
    all_moves = get_all_possible_moves(mon)
    return bool(all_moves & PRIORITY_MOVES)

def has_hazard_moves(mon: Any) -> bool:
    """
    Check if Pokemon can set hazards.
    
    Args:
        mon: Pokemon object
    
    Returns:
        True if has hazard moves in movepool
    """
    all_moves = get_all_possible_moves(mon)
    return bool(all_moves & HAZARD_MOVES)

def has_removal_moves(mon: Any) -> bool:
    """
    Check if Pokemon can remove hazards.
    
    Args:
        mon: Pokemon object
    
    Returns:
        True if has removal moves in movepool
    """
    all_moves = get_all_possible_moves(mon)
    return bool(all_moves & REMOVAL_MOVES)

def get_possible_items(mon: Any) -> List[str]:
    """
    Get list of possible items this Pokemon could have.
    
    Args:
        mon: Pokemon object
    
    Returns:
        List of item names
    """
    data = get_species_data(mon)
    if not data:
        return []
    
    return data.get('items', [])

def can_have_choice_item(mon: Any) -> bool:
    """Check if Pokemon could have a Choice item."""
    items = get_possible_items(mon)
    return any('Choice' in item for item in items)

def can_have_heavy_duty_boots(mon: Any) -> bool:
    """Check if Pokemon could have Heavy-Duty Boots."""
    items = get_possible_items(mon)
    return 'Heavy-Duty Boots' in items

def can_have_leftovers(mon: Any) -> bool:
    """Check if Pokemon could have Leftovers."""
    items = get_possible_items(mon)
    return 'Leftovers' in items

def estimate_max_damage_from_movepool(
    attacker: Any,
    defender: Any,
    battle: Any,
    dmg_fn,
) -> float:
    """
    Estimate maximum possible damage attacker could do with ANY possible move.
    
    This is useful for:
    - Evaluating setup safety (can opponent OHKO me with any move?)
    - Opponent threat assessment
    
    Args:
        attacker: Attacking Pokemon
        defender: Defending Pokemon
        battle: Battle object
        dmg_fn: Damage calculation function
    
    Returns:
        Maximum damage fraction (0.0 to 1.0+)
    """
    all_moves = get_all_possible_moves(attacker)
    
    # If we don't have movepool data, use known moves
    if not all_moves:
        max_dmg = 0.0
        for move in (getattr(attacker, 'moves', None) or {}).values():
            try:
                dmg = float(dmg_fn(move, attacker, defender, battle))
                max_dmg = max(max_dmg, dmg)
            except:
                pass
        return max_dmg
    
    # We have movepool - but we don't have actual move objects
    # This is a limitation: we'd need to create move objects from names
    # For now, return a conservative estimate based on known moves
    max_dmg = 0.0
    for move in (getattr(attacker, 'moves', None) or {}).values():
        try:
            dmg = float(dmg_fn(move, attacker, defender, battle))
            max_dmg = max(max_dmg, dmg)
        except:
            pass
    
    # If Pokemon has setup moves and we're checking threat, add bonus
    if has_setup_potential(attacker):
        max_dmg *= 1.3  # Assume potential +1/+2 boost
    
    return max_dmg


def get_archetype_summary(mon: Any) -> Dict[str, Any]:
    """
    Get a complete archetype summary for a Pokemon.
    
    Useful for debugging and understanding what the system knows.
    
    Args:
        mon: Pokemon object
    
    Returns:
        Dictionary with archetype information
    """
    return {
        'species': getattr(mon, 'species', 'Unknown'),
        'roles': get_role_names(mon),
        'is_physical': is_physical_attacker(mon),
        'is_fast': is_fast_sweeper(mon),
        'is_defensive': is_defensive(mon),
        'has_setup': has_setup_potential(mon),
        'has_priority': has_priority_moves(mon),
        'has_hazards': has_hazard_moves(mon),
        'has_removal': has_removal_moves(mon),
        'possible_items': get_possible_items(mon),
        'possible_moves': sorted(get_all_possible_moves(mon)),
    }