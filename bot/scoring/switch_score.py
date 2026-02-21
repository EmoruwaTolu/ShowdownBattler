from typing import Any

from poke_env.battle import PokemonType, Status

from bot.model.ctx import EvalContext
from bot.scoring.helpers import hp_frac, safe_types
from bot.scoring.damage_score import estimate_damage_fraction

SPIKES_DAMAGE = {1: 1.0 / 8.0, 2: 1.0 / 6.0, 3: 1.0 / 4.0}


def score_switch(pokemon: Any, battle: Any, ctx: EvalContext) -> float:
    """
    Score how good switching to `pokemon` is given the current battle state.

    Args:
        pokemon: The bench Pokemon being considered for switch-in
        battle: Current battle state
        ctx: EvalContext (ctx.me = current active, ctx.opp = opponent's active)

    Returns:
        Float score (higher = better switch)
    """
    opp = ctx.opp
    me = ctx.me

    if pokemon is None or opp is None:
        return -100.0

    switch_hp = hp_frac(pokemon)
    if switch_hp <= 0.0:
        return -200.0

    score = 0.0

    # Bringing in a low-HP mon is risky; healthy mons handle pressure better
    if switch_hp < 0.25:
        score -= 25.0
    elif switch_hp < 0.5:
        score -= 8.0
    elif switch_hp > 0.75:
        score += 5.0

    # Cost of switching through our side's entry hazards
    my_sc = _get_side_conditions(battle, our_side=True)
    score -= _hazard_penalty(pokemon, my_sc)

    # How hard can the opponent hit our switch-in?
    opp_best_damage = _best_opponent_damage(opp, pokemon, battle)
    score -= opp_best_damage * 60.0

    # Resistance/immunity bonus
    if opp_best_damage < 0.05:
        score += 40.0   # Immune or 4x resistant — great switch-in
    elif opp_best_damage <= 0.15:
        score += 20.0   # 2x resistant
    elif opp_best_damage < 0.25:
        score += 8.0    # Mildly resistant

    # Survival check: will the switch-in be KO'd on entry?
    # Only matters when we're switching voluntarily (active not fainted)
    active_fainted = (me is None) or (hp_frac(me) <= 0.0)
    if not active_fainted:
        if opp_best_damage >= switch_hp:
            score -= 50.0   # KO'd coming in — terrible
        elif opp_best_damage >= switch_hp * 0.70:
            score -= 20.0   # Left dangerously low

    # How hard can our switch-in hit the opponent back?
    my_best_damage = _best_offensive_damage(pokemon, opp, battle)
    score += my_best_damage * 40.0

    opp_hp = hp_frac(opp)
    if my_best_damage >= opp_hp:
        score += 25.0   # Can KO immediately after switching in
    elif my_best_damage >= opp_hp * 0.5:
        score += 8.0    # Strong 2HKO pressure

    score -= _status_penalty(pokemon)

    return float(score)

def _get_side_conditions(battle: Any, our_side: bool) -> dict:
    """Return side conditions as a normalized lowercase string-keyed dict."""
    try:
        if our_side:
            raw = getattr(battle, 'side_conditions', {}) or {}
        else:
            raw = getattr(battle, 'opponent_side_conditions', {}) or {}
        result = {}
        for key, val in raw.items():
            name = getattr(key, 'name', str(key)).lower().replace('_', '')
            result[name] = int(val) if val else 1
        return result
    except Exception:
        return {}


def _hazard_penalty(pokemon: Any, sc: dict) -> float:
    """Point penalty for switching through our side's entry hazards."""
    # Heavy-Duty Boots ignores all hazards
    item = str(getattr(pokemon, 'item', '') or '').lower().replace(' ', '').replace('-', '')
    if item == 'heavydutyboots':
        return 0.0

    penalty = 0.0
    grounded = _is_grounded(pokemon)

    # Stealth Rock — damage depends on type effectiveness vs Rock
    if sc.get('stealthrock', 0) > 0:
        dmg_frac = _sr_damage_frac(pokemon)
        penalty += dmg_frac * 40.0  # neutral (12.5%) → 5pts, 4x weak (50%) → 20pts

    # Spikes — grounded mons only
    spikes = min(3, max(0, sc.get('spikes', 0)))
    if spikes > 0 and grounded:
        penalty += SPIKES_DAMAGE[spikes] * 30.0

    # Toxic Spikes — grounded, non-Poison mons
    tspikes = min(2, max(0, sc.get('toxicspikes', 0)))
    if tspikes > 0 and grounded and not _is_poison_type(pokemon):
        penalty += 8.0 if tspikes >= 2 else 5.0  # TOX > PSN

    # Sticky Web — grounded; -1 Speed hurts sweepers
    if sc.get('stickyweb', 0) > 0 and grounded:
        penalty += 5.0

    return penalty


def _sr_damage_frac(pokemon: Any) -> float:
    """Stealth Rock damage as fraction of max HP for this Pokemon."""
    try:
        from poke_env.data import GenData
        type_chart = GenData.from_gen(9).type_chart
        mult = 1.0
        for t in (getattr(pokemon, 'type_1', None), getattr(pokemon, 'type_2', None)):
            if t is not None:
                mult *= PokemonType.damage_multiplier(PokemonType.ROCK, t, type_chart=type_chart)
        return (1.0 / 8.0) * float(mult)
    except Exception:
        return 0.125  # neutral fallback


def _is_grounded(pokemon: Any) -> bool:
    """True if the Pokemon is subject to Spikes / Sticky Web."""
    try:
        t1 = getattr(pokemon, 'type_1', None)
        t2 = getattr(pokemon, 'type_2', None)
        if t1 == PokemonType.FLYING or t2 == PokemonType.FLYING:
            return False
        ability = str(getattr(pokemon, 'ability', '') or '').lower().replace(' ', '').replace('-', '')
        if ability == 'levitate':
            return False
        item = str(getattr(pokemon, 'item', '') or '').lower().replace(' ', '').replace('-', '')
        if item == 'airballoon':
            return False
        return True
    except Exception:
        return True


def _is_poison_type(pokemon: Any) -> bool:
    try:
        t1 = getattr(pokemon, 'type_1', None)
        t2 = getattr(pokemon, 'type_2', None)
        return t1 == PokemonType.POISON or t2 == PokemonType.POISON
    except Exception:
        return False


def _best_opponent_damage(opp: Any, pokemon: Any, battle: Any) -> float:
    """Max damage the opponent can deal to our switch-in (as HP fraction)."""
    known_moves = getattr(opp, 'moves', {}) or {}
    best = 0.0
    for move in known_moves.values():
        try:
            dmg = float(estimate_damage_fraction(move, opp, pokemon, battle))
            best = max(best, dmg)
        except Exception:
            pass

    # Fallback to type inference when moves are unknown
    if best == 0.0:
        best = _type_fallback_damage(opp, pokemon)

    return best


def _best_offensive_damage(pokemon: Any, opp: Any, battle: Any) -> float:
    """Max damage our switch-in can deal to the opponent (as HP fraction)."""
    known_moves = getattr(pokemon, 'moves', {}) or {}
    best = 0.0
    for move in known_moves.values():
        try:
            dmg = float(estimate_damage_fraction(move, pokemon, opp, battle))
            best = max(best, dmg)
        except Exception:
            pass
    return best


def _type_fallback_damage(attacker: Any, defender: Any) -> float:
    """
    Estimate attacker's damage from type matchup when moves are unknown.
    Assumes the attacker uses their best STAB type.
    """
    try:
        from poke_env.data import GenData
        type_chart = GenData.from_gen(9).type_chart
        att_types = safe_types(attacker)
        def_types = safe_types(defender)
        if not att_types or not def_types:
            return 0.35

        best_mult = 0.0
        for att_type in att_types:
            if att_type is None:
                continue
            mult = 1.0
            for def_type in def_types:
                if def_type is None:
                    continue
                mult *= PokemonType.damage_multiplier(att_type, def_type, type_chart=type_chart)
            best_mult = max(best_mult, mult)

        # ~30% HP at neutral effectiveness; scales with type advantage
        return 0.30 * best_mult
    except Exception:
        return 0.30


def _status_penalty(pokemon: Any) -> float:
    """Point penalty for an already-existing status on the switch-in."""
    status = getattr(pokemon, 'status', None)
    if status is None:
        return 0.0
    if status == Status.PAR:
        return 8.0
    if status == Status.BRN:
        return 10.0
    if status in (Status.PSN, Status.TOX):
        return 6.0
    if status in (Status.SLP, Status.FRZ):
        return 15.0
    return 0.0
