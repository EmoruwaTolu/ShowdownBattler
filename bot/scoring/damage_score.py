from __future__ import annotations
from typing import Any, Optional, Tuple
from poke_env.calc.damage_calc_gen9 import calculate_damage
from poke_env.battle import MoveCategory

def _safe_species(p: Any) -> str:
    s = str(getattr(p, "species", "") or "").strip().lower()
    # Sometimes poke-env has `species` vs `name` vs `pokemon_id`
    if not s:
        s = str(getattr(p, "name", "") or "").strip().lower()
    return s

def _safe_hp_frac(p: Any) -> Optional[float]:
    try:
        hp = getattr(p, "current_hp", None)
        mx = getattr(p, "max_hp", None)
        if hp is None or mx is None or mx <= 0:
            # some mocks only have `hp` inside stats
            # if no reliable hp, return None
            return None
        return float(hp) / float(mx)
    except Exception:
        return None

def _get_pokemon_identifier(pokemon: Any, battle: Any) -> Optional[str]:
    """
    Robustly get battle identifier for a Pokemon (e.g., "p1: Gengar", "p2: Zacian").

    Strategy:
      1) identity match (exact object)
      2) species match within the same side (team / opponent_team), tie-break by HP fraction closeness
      3) if still ambiguous, return the first species match on either side (rare; better than None)
    """
    if pokemon is None or battle is None:
        return None

    # Build candidate pools
    team_items = []
    opp_items = []
    try:
        team_items = list((getattr(battle, "team", None) or {}).items())
    except Exception:
        team_items = []
    try:
        opp_items = list((getattr(battle, "opponent_team", None) or {}).items())
    except Exception:
        opp_items = []

    # 1) Exact identity match
    for identifier, pkmn in team_items:
        if pkmn is pokemon:
            return identifier
    for identifier, pkmn in opp_items:
        if pkmn is pokemon:
            return identifier

    # Helper: choose best match by species + hp closeness
    target_species = _safe_species(pokemon)
    target_hp = _safe_hp_frac(pokemon)

    def best_species_match(items: list[Tuple[str, Any]]) -> Optional[str]:
        # Collect all same-species candidates
        cands = [(ident, p) for ident, p in items if _safe_species(p) == target_species]
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0][0]

        # If we have hp info, choose closest hp fraction
        if target_hp is not None:
            scored = []
            for ident, p in cands:
                hp = _safe_hp_frac(p)
                if hp is None:
                    # unknown hp -> mild penalty so known hp wins
                    scored.append((1.0, ident))
                else:
                    scored.append((abs(hp - target_hp), ident))
            scored.sort(key=lambda t: t[0])
            return scored[0][1]

        # Otherwise, ambiguous: return first
        return cands[0][0]

    # 2) Prefer same-side match (if we can infer side from identifier-style, we can’t here)
    # But we can still try team then opponent (this is usually correct for switch-ins you evaluate)
    if target_species:
        ident = best_species_match(team_items)
        if ident is not None:
            return ident
        ident = best_species_match(opp_items)
        if ident is not None:
            return ident

    # 3) As a final fallback, if species missing, try matching by name/id fields (rare)
    # (If nothing works, return None)
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