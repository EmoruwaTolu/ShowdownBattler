from __future__ import annotations
import math
from typing import Any, Optional
from poke_env.calc.damage_calc_gen9 import calculate_damage
from poke_env.battle import MoveCategory
from poke_env.data import GenData

_GEN9_TYPE_CHART = None

def _get_type_chart():
    global _GEN9_TYPE_CHART
    if _GEN9_TYPE_CHART is None:
        _GEN9_TYPE_CHART = GenData.from_gen(9).type_chart
    return _GEN9_TYPE_CHART


def _stat_estimate(base: int, evs: int = 0, ivs: int = 31, level: int = 100) -> int:
    """Estimate a non-HP stat from base stat (Gen 9 formula)."""
    return math.floor((2 * base + ivs + evs // 4) * level / 100) + 5


def _hp_estimate(base: int, evs: int = 0, ivs: int = 31, level: int = 100) -> int:
    """Estimate HP stat from base stat (Gen 9 formula)."""
    return math.floor((2 * base + ivs + evs // 4) * level / 100) + level + 10


def _smart_damage_fallback(move: Any, me: Any, opp: Any) -> float:
    """
    Estimate damage fraction using base stats + STAB + type effectiveness when
    poke_env's calculate_damage fails (e.g. opponent stats not yet populated).

    Uses the Gen 9 damage formula with:
    - Attacker's actual stats if available (our Pokemon), else estimated from base stats
    - Defender's stats estimated from base stats (conservative: 0 EVs)
    - Defender's HP estimated from base stats for the denominator

    This ensures STAB and type-effective moves score much higher than the flat
    0.25 fallback, fixing move-ordering issues for moves like Hydropump.
    """
    bp = float(getattr(move, 'base_power', 0) or 0)
    if bp <= 0:
        return 0.0

    is_physical = getattr(move, 'category', None) == MoveCategory.PHYSICAL
    atk_key = 'atk' if is_physical else 'spa'
    def_key = 'def' if is_physical else 'spd'

    # Attacker offensive stat: use actual stat if available (our own Pokemon)
    me_stats = getattr(me, 'stats', {}) or {}
    attack = me_stats.get(atk_key)
    if not isinstance(attack, (int, float)) or attack is None:
        me_base = getattr(me, '_base_stats', {}) or {}
        attack = _stat_estimate(int(me_base.get(atk_key, 100) or 100))

    # Defender defensive stat: always estimate from base stats
    opp_base = getattr(opp, '_base_stats', {}) or {}
    defense = _stat_estimate(int(opp_base.get(def_key, 100) or 100))

    # Defender max HP: use actual if it looks real (> 100), else estimate
    opp_max_hp = float(getattr(opp, 'max_hp', 0) or 0)
    if opp_max_hp <= 100:
        opp_max_hp = float(_hp_estimate(int(opp_base.get('hp', 100) or 100)))

    attack = max(1.0, float(attack))
    defense = max(1.0, float(defense))

    # Gen 9 damage formula (level 100)
    base_dmg = math.floor(math.floor(22 * bp * attack / defense / 50) + 2)

    # STAB multiplier
    move_type = getattr(move, 'type', None)
    me_types = getattr(me, 'types', []) or []
    stab = 1.5 if (move_type is not None and move_type in me_types) else 1.0

    # Type effectiveness
    type_eff = 1.0
    if move_type is not None:
        try:
            tc = _get_type_chart()
            opp_types = getattr(opp, 'types', []) or []
            t1 = opp_types[0] if len(opp_types) > 0 else None
            t2 = opp_types[1] if len(opp_types) > 1 else None
            if t1 is not None:
                type_eff = move_type.damage_multiplier(t1, t2, type_chart=tc)
        except Exception:
            pass

    # Average damage roll (92.5% of max = midpoint of 85%–100% range)
    avg_dmg = base_dmg * stab * type_eff * 0.925
    return avg_dmg / opp_max_hp

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
    Get battle identifier for a Pokemon (e.g., "p1: Gengar", "p2: Zacian").

    Uses object-identity matching only.  A species-name fallback would silently
    map sampled/unknown MCTS Pokemon to real team members on the wrong side, causing
    poke_env’s calculate_damage to call battle.get_pokemon("p1: Bellibolt") for what
    was actually our own bench Bellibolt — creating spurious entries in _opponent_team
    that inflate the seen-count and break MCTS unseen-slot tracking.
    """
    if pokemon is None or battle is None:
        return None

    try:
        for identifier, pkmn in (getattr(battle, "team", None) or {}).items():
            if pkmn is pokemon:
                return identifier
    except Exception:
        pass

    try:
        for identifier, pkmn in (getattr(battle, "opponent_team", None) or {}).items():
            if pkmn is pokemon:
                return identifier
    except Exception:
        pass

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

    # Ensure the Pokemon the calc will see have correct item (mock battles / lookups)
    try:
        att = battle.get_pokemon(me_identifier)
        def_ = battle.get_pokemon(opp_identifier)
        if att is not None and getattr(me, "item", None) is not None:
            att.item = (getattr(me, "item", None) or "").strip().lower().replace(" ", "").replace("-", "") or None
        if def_ is not None and getattr(opp, "item", None) is not None:
            def_.item = (getattr(opp, "item", None) or "").strip().lower().replace(" ", "").replace("-", "") or None
    except Exception:
        pass

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

        if opp_max_hp is None or opp_max_hp <= 0:
            return _smart_damage_fallback(move, me, opp)

        # Return average damage as fraction of HP
        avg_dmg = (min_dmg + max_dmg) / 2.0
        result = avg_dmg / opp_max_hp

        # Sanity-check: if the poke-env calc gives a wildly inflated result
        # (can happen when opp.max_hp is percentage-scale but damage is absolute),
        # fall back to the smart estimate instead.
        if result > 3.0:
            return _smart_damage_fallback(move, me, opp)

        return result

    except Exception as _dmg_exc:
        # calculate_damage failed (usually because opponent stats are not yet
        # populated for random-battle opponents).  Use the smart fallback which
        # estimates damage from base stats + STAB + type effectiveness.
        import os
        if os.environ.get('BATTLER_DEBUG_DMG'):
            import traceback
            print(f"[DMG_FALLBACK] {getattr(move,'id','?')} vs {getattr(opp,'species','?')}: {_dmg_exc}")
            traceback.print_exc()
        return _smart_damage_fallback(move, me, opp)


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