"""
Belief-based penalties for switch scoring.

Use beliefs for risk, not reward: expected unrevealed damage, tail risk (chunk/OHKO),
item-based swing (Scarf, Helmet), and free-progress probabilities (setup, pivot, hazards, status).
"""

from typing import Any, Dict, Optional

from bot.model.opponent_model import (
    OpponentBelief,
    build_opponent_belief,
    build_move_pool,
)
from bot.scoring.damage_score import estimate_damage_fraction

# Move categories for free-progress inference
_SETUP_MOVE_IDS = frozenset({
    'swordsdance', 'nastyplot', 'dragondance', 'calmmind', 'bulkup',
    'quiverdance', 'shellsmash', 'bellydrum', 'shiftgear', 'agility',
    'tailglow', 'coil', 'curse', 'growth',
})
_HAZARD_MOVE_IDS = frozenset({'stealthrock', 'spikes', 'toxicspikes', 'stickyweb'})
_STATUS_MOVE_IDS = frozenset({
    'toxic', 'poisonpowder', 'willowisp', 'thunderwave', 'glare', 'stunspore',
    'spore', 'sleeppowder', 'hypnosis', 'sing', 'yawn', 'lovelykiss', 'grasswhistle', 'darkvoid',
})
_PIVOT_MOVE_IDS = frozenset({
    'uturn', 'voltswitch', 'flipturn', 'partingshot', 'chillyreception', 'teleport',
})
_RECOVERY_MOVE_IDS = frozenset({
    'recover', 'softboiled', 'slackoff', 'roost', 'moonlight', 'morningsun',
    'synthesis', 'shoreup', 'leechseed', 'strengthsap', 'oblivionwing',
})

# Weights for belief penalties (use for risk, not reward)
_LAMBDA_COVERAGE = 25.0      # expected unrevealed damage
_A_CHUNK = 18.0              # P(best_unrevealed_dmg >= 0.60) — surprise risk
_B_OHKO = 30.0               # P(best_unrevealed_dmg >= 0.90)
_SCARF_PENALTY = 8.0         # speed surprise danger
_HELMET_PENALTY = 6.0        # pivot contact punishment

# Clamp: belief is overlay, not whole score
_MAX_BELIEF_PENALTY = 45.0   # cap so base score dominates
_CHOICE_DAMAGE_PENALTY = 4.0  # Band/Specs: turns "safe-ish" switch into huge punish


def _norm_id(s: Any) -> str:
    if isinstance(s, str):
        return s.lower().replace(' ', '').replace('-', '').replace('_', '')
    return str(getattr(s, 'id', getattr(s, 'name', '')) or '').lower().replace(' ', '').replace('-', '').replace('_', '')


def _is_attacking_move(move: Any) -> bool:
    """True if move can deal damage (not pure status)."""
    if move is None:
        return False
    try:
        from poke_env.battle import MoveCategory
        cat = getattr(move, 'category', None)
        if cat == MoveCategory.STATUS:
            return False
        bp = float(getattr(move, 'base_power', 0) or 0)
        return bp > 0
    except Exception:
        return False


def _normalized_dist(belief: Any):
    """Return (cand, prob/Z) pairs; Z = sum(prob). Handles unnormalized dist."""
    dist = getattr(belief, 'dist', None) or []
    if not dist:
        return []
    Z = sum(prob for _, prob in dist)
    if Z <= 0:
        return []
    return [(cand, prob / Z) for cand, prob in dist]


def _cand_has_setup(cand: Any) -> bool:
    """Robust setup check: use cand.has_setup when available, else check moves directly with normalization."""
    try:
        if getattr(cand, 'has_setup', False):
            return True
    except (AttributeError, TypeError):
        pass
    moves = getattr(cand, 'moves', set()) or set()
    return any(_norm_id(m) in _SETUP_MOVE_IDS for m in moves)


def belief_free_progress_probs(belief: Optional[OpponentBelief]) -> Dict[str, float]:
    """
    Compute P(opp has setup/pivot/status/hazards/recover) from belief distribution.
    Returns dict with p_setup, p_pivot, p_status, p_hazards, p_recover.
    Uses _norm_id for move matching to handle format/casing differences.
    """
    out = {'p_setup': 0.0, 'p_pivot': 0.0, 'p_status': 0.0, 'p_hazards': 0.0, 'p_recover': 0.0}
    dist = _normalized_dist(belief) if belief else []
    if not dist:
        return out

    for cand, prob in dist:
        try:
            if _cand_has_setup(cand):
                out['p_setup'] += prob
            moves = getattr(cand, 'moves', set()) or set()
            if any(_norm_id(m) in _PIVOT_MOVE_IDS for m in moves):
                out['p_pivot'] += prob
            if any(_norm_id(m) in _STATUS_MOVE_IDS for m in moves):
                out['p_status'] += prob
            if any(_norm_id(m) in _HAZARD_MOVE_IDS for m in moves):
                out['p_hazards'] += prob
            if any(_norm_id(m) in _RECOVERY_MOVE_IDS for m in moves):
                out['p_recover'] += prob
        except (AttributeError, TypeError, KeyError):
            pass

    return out


def belief_item_probs(belief: Optional[OpponentBelief]) -> Dict[str, float]:
    """Compute P(opp has Choice Scarf, Rocky Helmet, etc.) from belief distribution."""
    out = {
        'p_scarf': 0.0, 'p_band': 0.0, 'p_specs': 0.0,
        'p_helmet': 0.0, 'p_boots': 0.0,
    }
    dist = _normalized_dist(belief) if belief else []
    if not dist:
        return out

    for cand, prob in dist:
        items = getattr(cand, 'items', set()) or set()
        norm_items = {_norm_id(i) for i in items}
        if 'choicescarf' in norm_items:
            out['p_scarf'] += prob
        if 'choiceband' in norm_items:
            out['p_band'] += prob
        if 'choicespecs' in norm_items:
            out['p_specs'] += prob
        if 'rockyhelmet' in norm_items:
            out['p_helmet'] += prob
        if 'heavydutyboots' in norm_items:
            out['p_boots'] += prob

    return out


def belief_damage_terms(
    opp: Any,
    pokemon: Any,
    battle: Any,
    belief: Optional[OpponentBelief],
    move_pool: Optional[Dict[str, Any]],
) -> tuple:
    """
    Returns (expected_coverage_damage, p_chunk, p_ohko).

    - expected_coverage: E_s[ best_unrevealed_damage(s) ] into our switch-in
    - p_chunk: P_s( best_unrevealed_damage(s) >= 0.60 ) — surprise risk only
    - p_ohko: P_s( best_unrevealed_damage(s) >= 0.90 )
    """
    expected_cov = 0.0
    p_chunk = 0.0
    p_ohko = 0.0

    dist = _normalized_dist(belief) if belief else []
    if not dist or not move_pool:
        return (expected_cov, p_chunk, p_ohko)

    revealed = getattr(belief, 'revealed_moves', set()) or set()
    revealed_norm = {_norm_id(m) for m in revealed}

    for cand, prob in dist:
        best_damage = 0.0
        best_unrevealed = 0.0

        for mid in cand.moves:
            mn = _norm_id(mid)
            move_obj = move_pool.get(mid) or move_pool.get(mn)
            if move_obj is None:
                continue
            if not _is_attacking_move(move_obj):
                continue
            try:
                dmg = float(estimate_damage_fraction(move_obj, opp, pokemon, battle))
                best_damage = max(best_damage, dmg)
                if mn not in revealed_norm:
                    best_unrevealed = max(best_unrevealed, dmg)
            except Exception:
                pass

        expected_cov += prob * best_unrevealed
        # Tail risk = surprise risk (unrevealed moves only), not overall
        if best_unrevealed >= 0.90:
            p_ohko += prob
        if best_unrevealed >= 0.60:
            p_chunk += prob

    return (expected_cov, p_chunk, p_ohko)


def _game_phase_scale(battle: Any) -> float:
    """
    Belief penalties scale by game phase.
    Early (high uncertainty): 1.2
    Mid: 1.0
    Late (low uncertainty): 0.7
    """
    try:
        turn = int(getattr(battle, 'turn', 0) or 0)
    except Exception:
        return 1.0

    if turn < 8:
        return 1.2
    if turn >= 15:
        return 0.7
    return 1.0


def belief_penalties_total(
    opp: Any,
    pokemon: Any,
    battle: Any,
    effective_hp: float,
    likely_to_pivot: bool,
    belief: Optional[OpponentBelief] = None,
) -> float:
    """
    Total belief-based penalty for switching to pokemon.
    Use beliefs for risk only: coverage, tail risk, item swing.
    likely_to_pivot: only apply Helmet penalty when pivoting is likely (safe entry + contact pivot).
    Pass belief to avoid rebuilding when evaluating multiple switch candidates.
    """
    try:
        gen = int(getattr(getattr(battle, 'format', None), 'gen', 9) or 9)
    except Exception:
        gen = 9

    if belief is None:
        belief = build_opponent_belief(opp, gen)
    move_pool = build_move_pool(belief, gen)

    penalty = 0.0

    # Expected coverage damage (unrevealed moves)
    expected_cov, p_chunk, p_ohko = belief_damage_terms(
        opp, pokemon, battle, belief, move_pool
    )
    penalty += _LAMBDA_COVERAGE * expected_cov

    # Tail risk (surprise risk: unrevealed moves only)
    penalty += _A_CHUNK * p_chunk + _B_OHKO * p_ohko

    # Item-based swing
    item_probs = belief_item_probs(belief)
    # Scarf: proxy for "relying on moving first / frailty / speed assumptions"
    if effective_hp < 0.7:
        penalty += _SCARF_PENALTY * item_probs['p_scarf']
    # Band/Specs: turns "safe-ish" switch into huge punish when we're fragile
    if effective_hp < 0.65:
        penalty += _CHOICE_DAMAGE_PENALTY * (
            item_probs['p_band'] + item_probs['p_specs']
        )
    # Helmet: only when we're likely to click U-turn/Flip Turn (safe enough to pivot)
    if likely_to_pivot:
        penalty += _HELMET_PENALTY * item_probs['p_helmet']

    # Scale by game phase (early 1.2, mid 1.0, late 0.7)
    penalty *= _game_phase_scale(battle)

    # Clamp: overlay not whole score
    penalty = min(penalty, _MAX_BELIEF_PENALTY)

    return penalty
