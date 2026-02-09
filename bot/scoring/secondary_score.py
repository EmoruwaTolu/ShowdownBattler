from __future__ import annotations
from typing import Any, Optional, List, Dict

from poke_env.battle import Status

from bot.model.ctx import EvalContext
from bot.scoring.status_score import (
    burn_immediate_value,
    para_immediate_value,
    poison_immediate_value,
    sleep_immediate_value,
    freeze_immediate_value,
    status_is_applicable,
)


def score_secondaries(move: Any, battle: Any, ctx: EvalContext, ko_prob: float, dmg_frac: Optional[float] = None) -> float:
    """
    Score secondary status effects on damage moves.
    
    Examples:
    - Scald (80 BP Water, 30% burn)
    - Thunderbolt (90 BP Electric, 10% para)
    - Ice Beam (90 BP Ice, 10% freeze)
    
    Formula for each secondary:
        EV = chance × accuracy × (1 - ko_prob) × status_value
    
    Args:
        move: The damage move with secondary effects
        battle: Current battle state
        ctx: Evaluation context
        ko_prob: Probability this move KOs the opponent
        dmg_frac: Damage fraction (not used, kept for compatibility)
    
    Returns:
        Total expected value from all secondary effects (0-40 points)
        
    Example:
        Scald vs physical attacker:
        - Secondary: 30% burn
        - Status value: 45 (burn vs physical)
        - Accuracy: 100%
        - KO prob: 0%
        - EV = 0.30 × 1.0 × 1.0 × 45 = 13.5 points
    """
    me = ctx.me
    opp = ctx.opp
    
    if me is None or opp is None:
        return 0.0
    
    # Can't inflict status if opponent already has one
    if getattr(opp, 'status', None) is not None:
        return 0.0
    
    # Get secondary effects
    secondaries = getattr(move, 'secondary', None) or []
    if not secondaries:
        return 0.0
    
    # If we KO, status doesn't matter (they faint before it procs)
    no_ko_prob = max(0.0, 1.0 - ko_prob)
    
    # Get move accuracy
    accuracy = getattr(move, 'accuracy', 1.0) or 1.0
    accuracy = max(0.0, min(1.0, accuracy))
    
    total_value = 0.0
    
    for sec in secondaries:
        # Parse status from secondary
        status = parse_secondary_status(sec)
        if status is None:
            continue
        
        # Check if status can be applied (type immunities)
        if not status_is_applicable(status, move, opp):
            continue
        
        # Get secondary chance (e.g., 30% for Scald)
        chance = get_secondary_chance(sec)
        if chance <= 0:
            continue
        
        # Get status value, it will be the same as pure status moves!
        status_value = get_status_value(status, me, opp)
        
        # Expected value formula:
        # EV = chance × accuracy × (1 - ko_prob) × status_value
        ev = chance * accuracy * no_ko_prob * status_value
        
        total_value += ev
    
    return total_value

def parse_secondary_status(sec: Dict[str, Any]) -> Optional[Status]:
    """Parse status from secondary effect dict."""
    status_name = sec.get('status', None)
    if not status_name:
        return None
    
    try:
        return Status[status_name.upper()]
    except:
        return None


def get_secondary_chance(sec: Dict[str, Any]) -> float:
    """Get chance of secondary effect proccing (0.0 to 1.0)."""
    if 'chance' not in sec:
        return 1.0  # 100% if not specified
    
    try:
        chance = float(sec['chance']) / 100.0
        return max(0.0, min(1.0, chance))
    except:
        return 0.0


def get_status_value(status: Status, me: Any, opp: Any) -> float:
    """
    Get status value using the SAME functions as status_score.py.
    
    This ensures consistency:
    - Scald's 30% burn worth exactly 30% of Will-O-Wisp's value
    - Thunderbolt's 10% para worth exactly 10% of Thunder Wave's value
    
    Returns:
        Status value in points (same scale as pure status moves)
    """
    if status == Status.BRN:
        return burn_immediate_value(me, opp)
    
    elif status == Status.PAR:
        return para_immediate_value(me, opp)
    
    elif status in (Status.PSN, Status.TOX):
        return poison_immediate_value(status)
    
    elif status == Status.SLP:
        return sleep_immediate_value()
    
    elif status == Status.FRZ:
        return freeze_immediate_value()
    
    return 20.0  # Default for unknown status