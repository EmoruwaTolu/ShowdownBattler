from typing import Any, Set, Tuple
from bot.model.opponent_model import get_opponent_set_distribution, physical_prob


def hp_frac(p: Any) -> float:
    try:
        return float(p.current_hp_fraction)
    except Exception:
        return 0.0


def is_fainted(p: Any) -> bool:
    try:
        return bool(p.fainted)
    except Exception:
        return False


def remaining_count(team: dict) -> int:
    return sum(1 for p in team.values() if p and not is_fainted(p))


def safe_types(p: Any) -> Set[Any]:
    """
    poke-env 0.8.x usually exposes pokemon.types as a tuple of PokemonType.
    """
    try:
        return set(p.types or [])
    except Exception:
        return set()


def looks_like_physical_attacker(p: Any) -> bool:
    """
    Abstract fallback heuristic: base atk significantly higher than base spa.
    Works even without knowing moves.
    """
    try:
        bs = p.base_stats or {}
        atk = bs.get("atk", 100)
        spa = bs.get("spa", 100)
        return atk >= spa * 1.15
    except Exception:
        return False


def physical_probability(opp: Any, battle: Any, ctx: Any, default_gen: int = 9) -> float:
    """
    Returns P(opponent is physical attacker | candidate randbats sets).
    Falls back to looks_like_physical_attacker if we can't get a distribution.

    Output is in [0, 1].
    """
    if opp is None:
        return 0.0

    # figure out gen
    gen = getattr(getattr(battle, "format", None), "gen", None)
    if gen is None:
        gen = getattr(getattr(ctx, "battle", None), "gen", default_gen) or default_gen

    # Lazy import (prevents circular imports)
    try:
        dist = get_opponent_set_distribution(opp, 9)
        if dist:
            p = float(physical_prob(dist))
            return max(0.0, min(1.0, p))
    except Exception:
        pass

    # fallback if we can't find a moveset for the mon in the current randbats database
    try:
        return 1.0 if looks_like_physical_attacker(opp) else 0.0
    except Exception:
        return 0.5


def looks_like_setup_sweeper(p: Any) -> bool:
    """
    For *your own* team, you often know moves (randbats reveals your set).
    """
    try:
        for m in p.moves.values():
            if getattr(m, "boosts", None) or getattr(m, "self_boost", None):
                return True
    except Exception:
        pass
    return False


def is_slower(me: Any, opp: Any) -> bool:
    """
    Rough speed compare using base speed (no item/para/boost modeling).
    """
    try:
        ms = (me.base_stats or {}).get("spe", 80)
        os = (opp.base_stats or {}).get("spe", 80)
        return ms < os
    except Exception:
        return False


def ally_has_priority(ally: Any) -> bool:
    try:
        for m in (ally.moves or {}).values():
            if getattr(m, "priority", 0) > 0:
                return True
    except Exception:
        pass
    return False


def ally_is_frail(ally: Any) -> bool:
    """
    Rough bulk score (lower => frailer).
    """
    try:
        bs = ally.base_stats or {}
        hp = bs.get("hp", 80)
        d = bs.get("def", 80)
        sd = bs.get("spd", 80)
        bulk = hp + d + sd
        return bulk < 240
    except Exception:
        return False

def opponent_set_probs(opp: Any, battle: Any, ctx: Any, default_gen: int = 9) -> Tuple[float, float, float]:
    """
    Returns (phys_p, setup_p, priority_p) under opponent candidate-set uncertainty.
    Falls back gracefully.
    """
    if opp is None:
        return 0.0, 0.0, 0.0

    gen = getattr(getattr(battle, "format", None), "gen", None)
    if gen is None:
        gen = getattr(getattr(ctx, "battle", None), "gen", default_gen) or default_gen

    try:
        dist = get_opponent_set_distribution(opp, int(gen))
        if dist:
            phys_p = float(physical_prob_from_dist(dist))
            setup_p = float(sum(w for c, w in dist if c.has_setup))
            prio_p = float(sum(w for c, w in dist if c.has_priority))
            return clamp(phys_p), clamp(setup_p), clamp(prio_p)
    except Exception:
        pass

    try:
        phys_p = 1.0 if looks_like_physical_attacker(opp) else 0.0
    except Exception:
        phys_p = 0.5
    return clamp(phys_p), 0.0, 0.0


def physical_prob_from_dist(dist):
    return physical_prob(dist)

def clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))