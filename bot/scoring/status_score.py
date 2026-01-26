from typing import Any, Optional
import math

# poke-env 0.11.0 imports - classes are in poke_env.battle module
from poke_env.battle import MoveCategory
from poke_env.battle import Status
from poke_env.battle import PokemonType

# Try to import damage calculator (available in poke-env 0.10.0+)
try:
    from poke_env.calc.damage_calc_gen9 import calculate_damage
    HAS_DAMAGE_CALC = True
except ImportError:
    HAS_DAMAGE_CALC = False
    print("WARNING: poke-env damage calculator not available (requires 0.10.0+)")
    print("Falling back to pressure-based damage estimation")

from bot.model.ctx import EvalContext
from bot.scoring.helpers import (
    hp_frac,
    remaining_count,
    safe_types,
    looks_like_setup_sweeper,
    is_slower,
    physical_probability,
    ally_has_priority,
    ally_is_frail,
)
from bot.scoring.damage_score import estimate_damage_fraction
from bot.model.opponent_model import get_opponent_set_distribution
from bot.scoring.race import evaluate_race_for_move
from bot.scoring.pressure import estimate_opponent_pressure


# ============================================================================
# STATUS SCORING CONSTANTS
# ============================================================================

# Base status values (intrinsic power of each status)
BASE_STATUS_SLEEP = 55.0
BASE_STATUS_PARALYSIS = 40.0
BASE_STATUS_BURN = 38.0
BASE_STATUS_POISON = 32.0
BASE_STATUS_DEFAULT = 20.0

# Burn chip damage (1/16 HP per turn)
BURN_CHIP_PER_TURN = 1.0 / 16.0
BURN_CHIP_VALUE_MULTIPLIER = 85.0

# Miss cost base and scaling
MISS_COST_BASE = 18.0
MISS_COST_DAMAGED_SCALE = 18.0
MISS_COST_SLOWER_PENALTY = 8.0
MISS_COST_BOOSTED_OPP_PENALTY = 10.0
MISS_COST_VALUE_SCALE = 0.20  # Miss cost scales 20% with status value

# Tempo risk
TEMPO_RISK_BASE = 12.0
TEMPO_RISK_DAMAGED_SCALE = 18.0
TEMPO_RISK_SLOWER_PENALTY = 6.0
TEMPO_RISK_BOOSTED_OPP_PENALTY = 6.0

# Race state modifiers (now scale with race degree)
RACE_BASE_MODIFIER = 18.0  # Base amount for slight win/loss

# Team synergy
SYNERGY_WINCON_SETUP_MULT = 1.35
SYNERGY_WINCON_BULK_MULT = 1.10
SYNERGY_SECOND_BEST_WEIGHT = 0.5


# ============================================================================
# APPLICABILITY CHECKS
# ============================================================================

def move_inflicts_major_status(move: Any) -> Optional[Status]:
    st = getattr(move, "status", None)
    return st if st is not None else None


def major_status_is_applicable(status: Status, move: Any, opp: Any) -> bool:
    if opp is None:
        return False

    if getattr(opp, "status", None) is not None:
        return False

    opp_types = safe_types(opp)

    if status == Status.BRN:
        if PokemonType.FIRE in opp_types:
            return False
    elif status in (Status.PSN, Status.TOX):
        if PokemonType.STEEL in opp_types or PokemonType.POISON in opp_types:
            return False
    elif status == Status.PAR:
        if PokemonType.ELECTRIC in opp_types:
            return False
        if getattr(move, "type", None) == PokemonType.ELECTRIC:
            if PokemonType.GROUND in opp_types:
                ignore = getattr(move, "ignore_immunity", False)
                if not ignore:
                    return False

    return True


def _best_damage_move(battle: Any) -> Optional[Any]:
    best = None
    best_exp = -1.0
    for mv in getattr(battle, "available_moves", []) or []:
        if getattr(mv, "category", None) == MoveCategory.STATUS:
            continue
        acc = float(getattr(mv, "accuracy", 1.0) or 1.0)
        acc = max(0.0, min(1.0, acc))
        exp = float(getattr(mv, "base_power", 0) or 0) * acc
        if exp > best_exp:
            best_exp = exp
            best = mv
    return best


def _race_state_if_we_just_attack(battle: Any, ctx: EvalContext) -> str:
    mv = _best_damage_move(battle)
    if mv is None:
        return "CLOSE"
    return evaluate_race_for_move(battle, ctx, mv).state


def status_miss_cost(battle: Any, ctx: EvalContext, pressure, status_value: float, accuracy: float) -> float:
    """
    Cost of missing a status move (proportional to value and accuracy).
    
    Miss cost increases when:
    - Status is more valuable (higher opportunity cost)
    - We're damaged (less time to recover)
    - We're slower (opponent gets free hit)
    - Opponent is boosted (dangerous to waste turn)
    - High pressure situations
    - Move has lower accuracy (higher risk)
    """
    me = ctx.me
    opp = ctx.opp

    # Base cost
    cost = MISS_COST_BASE
    
    # Scale with status value (higher value = worse to miss)
    cost += status_value * MISS_COST_VALUE_SCALE
    
    # Scale with miss chance (lower accuracy = worse penalty)
    # If accuracy is 50%, missing is not that surprising
    # If accuracy is 95%, missing feels terrible
    miss_chance = 1.0 - accuracy
    cost *= (1.0 + miss_chance * 0.5)  # Up to 50% more cost for low accuracy
    
    # Situation-dependent costs
    cost += (1.0 - hp_frac(me)) * MISS_COST_DAMAGED_SCALE

    if opp is not None and is_slower(me, opp):
        cost += MISS_COST_SLOWER_PENALTY

    try:
        if opp is not None and opp.boosts and any(v >= 2 for v in opp.boosts.values()):
            cost += MISS_COST_BOOSTED_OPP_PENALTY
    except Exception:
        pass

    cost += 8.0 * pressure.setup_prob + 6.0 * pressure.threat
    cost += 4.0 * pressure.physical_prob
    
    return cost

def score_status_move(move: Any, battle: Any, ctx: EvalContext) -> float:
    """
    Main status move scoring function.
    
    Components:
    1. Base status value (includes near/long-term via consolidated functions)
    2. Team synergy
    3. Tempo risk
    4. Race state modifier (scales with degree of winning/losing)
    5. Miss cost (scales with status value and accuracy)
    """
    opp = ctx.opp
    if opp is None:
        return -100.0

    major = move_inflicts_major_status(move)
    if major is not None and not major_status_is_applicable(major, move, opp):
        return -80.0

    # Cache pressure calculation
    cache_key = "opponent_pressure"
    if cache_key in ctx.cache:
        pressure = ctx.cache[cache_key]
    else:
        pressure = estimate_opponent_pressure(battle, ctx)
        ctx.cache[cache_key] = pressure

    raw = 0.0
    raw += base_status_value(move, battle, ctx)
    raw += team_synergy_value(move, battle, ctx)
    raw -= tempo_risk(move, battle, ctx, pressure)
    raw -= _survival_penalty(move, battle, ctx, pressure)

    # === IMPROVED RACE STATE MODIFIER ===
    # Scale modifier based on how much we're winning/losing
    
    mv = _best_damage_move(battle)
    if mv is not None:
        try:
            race = evaluate_race_for_move(battle, ctx, mv)
            race_state = race.state
            
            # Calculate degree of advantage/disadvantage
            tko_opp = getattr(race, "tko_opp", 99.0)
            ttd_me = getattr(race, "ttd_me", 99.0)
            
            if race_state == "WINNING":
                # We're winning: status less valuable
                # Scale by how much we're winning
                advantage = max(0.0, min(2.0, (ttd_me - tko_opp)))
                modifier = -RACE_BASE_MODIFIER * (advantage / 2.0)  # 0 to -18
                raw += modifier
            
            elif race_state == "LOSING":
                # We're losing: status more valuable (need time)
                # Scale by how much we're losing
                disadvantage = max(0.0, min(2.0, (tko_opp - ttd_me)))
                modifier = RACE_BASE_MODIFIER * (disadvantage / 2.0)  # 0 to +18
                raw += modifier
            
            # CLOSE stays at 0
        except Exception:
            pass

    # === ACCURACY WITH PROPORTIONAL MISS COST ===
    acc = float(getattr(move, "accuracy", 1.0) or 1.0)
    acc = max(0.0, min(1.0, acc))
    
    miss_cost = status_miss_cost(battle, ctx, pressure, raw, acc)
    final = acc * raw + (1.0 - acc) * (-miss_cost)
    
    return final


# ============================================================================
# CONSOLIDATED BURN SCORING
# ============================================================================

def _burn_total_value(battle: Any, ctx: EvalContext, pressure) -> float:
    """
    Consolidated burn value calculation - ALL burn logic in one place.
    
    Burn is valuable for:
    1. Halving physical attack damage (matchup dependent)
    2. Passive chip damage (1/16 HP per turn)
    3. Stopping physical setup sweepers
    
    This combines what was previously spread across:
    - base_status_value (base + special attacker penalty + chip)
    - near_term_payoff (physical matchup value + setup deterrent)
    - long_term_value (horizon scaling + ongoing effects)
    """
    opp = ctx.opp
    if opp is None:
        return 0.0
    
    # Start with base value
    value = BASE_STATUS_BURN
    
    phys_prob = pressure.physical_prob
    special_prob = 1.0 - phys_prob
    
    # === MATCHUP-SPECIFIC VALUE ===
    
    # Reduce value against special attackers (burn doesn't help much)
    value -= special_prob * 12.0
    
    # Near-term physical matchup value (next 1-3 turns)
    # Helps us survive physical attacks
    value += 8.0  # Base near-term
    value += 14.0 * phys_prob  # Strong vs physical
    value -= 2.0 * special_prob  # Weak vs special
    
    # === CHIP DAMAGE VALUE ===
    
    # Passive burn damage over expected game length
    our_remaining = remaining_count(battle.team)
    opp_remaining = remaining_count(battle.opponent_team)
    
    if opp_remaining >= 4 and our_remaining >= 4:
        expected_turns = 3.0
        horizon = 5  # long-term horizon
    elif opp_remaining >= 3:
        expected_turns = 2.5
        horizon = 3
    else:
        expected_turns = 2.0
        horizon = 2
    
    total_chip = BURN_CHIP_PER_TURN * expected_turns
    chip_value = total_chip * BURN_CHIP_VALUE_MULTIPLIER
    value += chip_value
    
    # === SETUP/PRESSURE DETERRENT ===
    
    # Burn stops physical setup sweepers (DD, SD, etc.)
    setup_value = 6.0 * pressure.setup_prob  # near-term
    setup_value += 5.0 * pressure.setup_prob  # long-term
    value += setup_value * phys_prob  # Only matters vs physical sweepers
    
    # === LONG-TERM VALUE ===
    
    # Ongoing benefit over multiple turns
    value += 3.5 * horizon
    value += 8.0 * phys_prob  # Additional long-term vs physical
    
    return value


def _burn_chip_damage_value(battle: Any, ctx: EvalContext) -> float:
    """
    Helper for secondary effects - just the chip damage portion.
    (Used by secondary_score.py)
    """
    opp = ctx.opp
    if opp is None:
        return 0.0
    
    our_remaining = remaining_count(battle.team)
    opp_remaining = remaining_count(battle.opponent_team)
    
    if opp_remaining >= 4 and our_remaining >= 4:
        expected_turns = 3.0
    elif opp_remaining >= 3:
        expected_turns = 2.5
    else:
        expected_turns = 2.0
    
    total_chip = BURN_CHIP_PER_TURN * expected_turns
    return total_chip * BURN_CHIP_VALUE_MULTIPLIER


# ============================================================================
# PARALYSIS SCORING
# ============================================================================

def _paralysis_speed_flip_value(me: Any, opp: Any, dist) -> float:
    """
    Helper: Calculate value of speed control from paralysis.
    
    Para HALVES opponent speed (Gen 7+), potentially flipping speed tiers.
    Returns value based on:
    - Whether we flip from slower to faster
    - Distribution of opponent speed multipliers (Choice Scarf, etc.)
    """
    if me is None or opp is None:
        return 0.0
    
    try:
        my_spe = (me.base_stats or {}).get("spe", 80)
        opp_base_spe = (opp.base_stats or {}).get("spe", 80)
    except Exception:
        my_spe, opp_base_spe = 80, 80
    
    # Calculate probability we become faster after para
    flip_prob = 0.0
    for cand, w in dist:
        eff_opp_spe = opp_base_spe * getattr(cand, "speed_mult", 1.0)
        if my_spe >= eff_opp_spe * 0.5:  # Para HALVES speed in Gen 7+
            flip_prob += w
    
    was_slower = my_spe < opp_base_spe
    
    # Major flip: we go from slower to faster
    if flip_prob > 0.8 and was_slower:
        return 22.0 * flip_prob
    # Minor improvement: already faster or uncertain flip
    elif flip_prob > 0.5:
        return 8.0 * flip_prob
    else:
        return 0.0


def _paralysis_total_value(battle: Any, ctx: EvalContext, pressure, dist) -> float:
    """
    Consolidated paralysis value calculation.
    
    Paralysis is valuable for:
    1. Speed control (flipping speed tiers)
    2. 25% full para chance (free turns)
    3. Stopping setup sweepers
    4. Long-term speed advantage
    """
    me = ctx.me
    opp = ctx.opp
    if opp is None or me is None:
        return 0.0
    
    # Base value
    value = BASE_STATUS_PARALYSIS
    
    slower = is_slower(me, opp)
    
    # === SPEED FLIP VALUE ===
    speed_flip_value = _paralysis_speed_flip_value(me, opp, dist)
    value += speed_flip_value
    
    # === NEAR-TERM PAYOFF ===
    
    # Base near-term value
    value += 10.0
    
    # Extra value if we're slower (speed control more important)
    if slower:
        value += 12.0
    else:
        value += 3.0
    
    # Setup deterrent (para helps vs setup sweepers)
    value += 10.0 * pressure.setup_prob
    
    # === LONG-TERM VALUE ===
    
    our_remaining = remaining_count(battle.team)
    opp_remaining = remaining_count(battle.opponent_team)
    
    if opp_remaining >= 4 and our_remaining >= 4:
        horizon = 5
    elif opp_remaining >= 3:
        horizon = 3
    else:
        horizon = 2
    
    value += 3.0 * horizon
    if slower:
        value += 6.0
    value += 6.0 * pressure.setup_prob
    
    return value


# ============================================================================
# OTHER STATUS VALUES
# ============================================================================

def _sleep_total_value() -> float:
    """Sleep value (very strong short-term, forces switch)."""
    # Near-term: 26.0
    # Long-term: 10.0
    # Base: 55.0
    return BASE_STATUS_SLEEP + 26.0 + 10.0


# ============================================================================
# ROLE PRESERVATION & SURVIVAL
# ============================================================================

def _survival_penalty(move: Any, battle: Any, ctx: EvalContext, pressure) -> float:
    """
    Penalty for clicking status when we're about to die and status won't help.
    
    If we're in OHKO range and status doesn't improve our survival,
    we should attack instead (deal damage before dying).
    
    Key insight: "Status into death" is almost always wrong unless the
    status helps us survive.
    """
    me = ctx.me
    opp = ctx.opp
    st = getattr(move, "status", None)
    
    if me is None or opp is None or st is None:
        return 0.0
    
    my_hp = hp_frac(me)
    opp_damage = pressure.damage_to_me_frac
    
    # Are we in OHKO range?
    if my_hp < opp_damage * 1.1:
        # We die next turn
        
        # Does this status help us survive?
        helps_survive = False
        
        if st == Status.BRN and pressure.physical_prob > 0.6:
            # Burn halves physical damage
            new_damage = opp_damage * 0.5
            if my_hp >= new_damage:
                helps_survive = True
        
        if st == Status.PAR:
            # Check if we become faster (speed flip)
            slower = is_slower(me, opp)
            if slower:
                try:
                    my_spe = (me.base_stats or {}).get("spe", 80)
                    opp_spe = (opp.base_stats or {}).get("spe", 80)
                    # After para, do we become faster?
                    if my_spe >= opp_spe * 0.5:
                        helps_survive = True
                except Exception:
                    pass
        
        # If status doesn't help us survive, heavy penalty
        if not helps_survive:
            # We're clicking status into death - this is bad!
            penalty = 40.0
            
            # Extra penalty if we're behind in Pokemon
            our_remaining = remaining_count(battle.team)
            opp_remaining = remaining_count(battle.opponent_team)
            if our_remaining < opp_remaining:
                penalty += (opp_remaining - our_remaining) * 10.0
            
            return penalty
    
    return 0.0


# ============================================================================
# MAIN STATUS VALUE FUNCTION
# ============================================================================

def base_status_value(move: Any, battle: Any = None, ctx: EvalContext = None) -> float:
    """
    Get the total value of a status condition.
    
    NOTE: This now returns the FULL value including near/long-term.
    Individual components (near_term_payoff, long_term_value) are deprecated
    but kept for backwards compatibility with secondary_score.py temporarily.
    """
    st = getattr(move, "status", None)
    
    if st is None:
        return BASE_STATUS_DEFAULT
    
    # For burn and paralysis, use consolidated functions if we have full context
    if battle is not None and ctx is not None:
        pressure = estimate_opponent_pressure(battle, ctx)
        
        if st == Status.BRN:
            return _burn_total_value(battle, ctx, pressure)
        
        if st == Status.PAR:
            gen = getattr(getattr(battle, "format", None), "gen", 9) or 9
            try:
                dist = get_opponent_set_distribution(ctx.opp, int(gen)) or []
            except Exception:
                dist = []
            return _paralysis_total_value(battle, ctx, pressure, dist)
    
    # Fallback to simple base values
    if st == Status.SLP:
        return _sleep_total_value()
    elif st == Status.PAR:
        return BASE_STATUS_PARALYSIS
    elif st == Status.BRN:
        return BASE_STATUS_BURN
    elif st in (Status.PSN, Status.TOX):
        if battle is not None and ctx is not None:
            # Use comprehensive poison model with all components
            return _poison_total_value(battle, ctx, pressure, move)
        else:
            return BASE_STATUS_POISON
    else:
        return BASE_STATUS_DEFAULT


def tempo_risk(move: Any, battle: Any, ctx: EvalContext, pressure = None) -> float:
    """
    Risk of losing tempo by clicking a status move instead of attacking.
    Now includes matchup-dependent penalties.
    """
    me = ctx.me
    opp = ctx.opp

    r = 12.0
    r += (1.0 - hp_frac(me)) * 18.0
    if opp is not None and is_slower(me, opp):
        r += 6.0

    try:
        if opp is not None and opp.boosts and any(v >= 1 for v in opp.boosts.values()):
            r += 6.0
    except Exception:
        pass
    
    # matchup-dependent penalty
    if pressure is not None:
        r += _status_matchup_penalty(move, battle, ctx, pressure)

    return r


def near_term_payoff(move: Any, battle: Any, ctx: EvalContext, pressure) -> float:
    """
    DEPRECATED: Now included in base_status_value via consolidated functions.
    
    Kept for backwards compatibility but returns 0.
    The consolidated _burn_total_value and _paralysis_total_value now include
    all near-term, long-term, and base values.
    """
    return 0.0


def long_term_value(move: Any, battle: Any, ctx: EvalContext, pressure) -> float:
    """
    DEPRECATED: Now included in base_status_value via consolidated functions.
    
    Kept for backwards compatibility but returns 0.
    """
    return 0.0


def team_synergy_value(move: Any, battle: Any, ctx: EvalContext) -> float:
    opp = ctx.opp
    if opp is None:
        return 0.0

    st = getattr(move, "status", None)
    if st is None:
        return 0.0

    gen = getattr(getattr(battle, "format", None), "gen", None)
    if gen is None:
        gen = getattr(getattr(ctx, "battle", None), "gen", 9) or 9

    try:
        dist = get_opponent_set_distribution(opp, 9) or []
    except Exception:
        dist = []

    phys_p = physical_probability(opp, battle, ctx)
    setup_p = sum(w for c, w in dist if getattr(c, "has_setup", False)) if dist else 0.25
    priority_p = sum(w for c, w in dist if getattr(c, "has_priority", False)) if dist else 0.15

    best = 0.0
    second = 0.0

    try:
        opp_base_spe = (opp.base_stats or {}).get("spe", 80)
    except Exception:
        opp_base_spe = 80

    for ally in battle.team.values():

        if ally is None:
            continue
        if getattr(ally, 'fainted', False):
            continue  
        if ally is ctx.me:
            continue
        
        setup_ally = looks_like_setup_sweeper(ally)
        priority_ally = ally_has_priority(ally)
        frail_ally = ally_is_frail(ally)

        try:
            ally_spe = (ally.base_stats or {}).get("spe", 80)
        except Exception:
            ally_spe = 80

        wincon_mult = 1.0
        if setup_ally:
            wincon_mult *= 1.35
        if not frail_ally:
            wincon_mult *= 1.10

        benefit = 0.0

        if st.name == "PAR":
            flip_prob = 0.0
            for cand, w in dist:
                eff_opp_spe = opp_base_spe * getattr(cand, "speed_mult", 1.0)
                if ally_spe >= eff_opp_spe * 0.5:  # Para HALVES speed in Gen 7+
                    flip_prob += w

            was_slower = ally_spe < opp_base_spe
            speed_flip_value = (22.0 if was_slower else 8.0) * flip_prob

            free_turn_ev = 0.25 * (14.0 + (10.0 if setup_ally else 0.0) + (8.0 if frail_ally else 0.0))
            stop_sweep = 18.0 * setup_p + 6.0 * (1.0 - flip_prob)
            if priority_p > 0 and frail_ally:
                stop_sweep += 6.0

            benefit += speed_flip_value + free_turn_ev + stop_sweep
            if priority_ally:
                benefit *= 0.85

        elif st.name == "BRN":
            # Get actual damage to ally using full pressure calculation
            damage_to_ally = _estimate_damage_to_ally(ally, opp, battle)
            
            # === KO THRESHOLD VALUE (NEW!) ===
            # Check if burn changes OHKO → 2HKO, 2HKO → 3HKO, etc.
            # This replaces the old generic dmg_reduction calculation
            ko_threshold_value = _burn_ko_threshold_value(
                ally, opp, battle, damage_to_ally, phys_p
            )
            
            # === SETUP DETERRENCE ===
            # This is independent of KO thresholds
            phys_setup_p = (
                sum(w for c, w in dist if getattr(c, "is_physical", False) and getattr(c, "has_setup", False))
                if dist else 0.20
            )
            stop_sweep = 26.0 * phys_setup_p + 8.0 * setup_p * phys_p
            stop_priority = 10.0 * priority_p * phys_p
            
            # COMBINE
            benefit += ko_threshold_value + stop_sweep + stop_priority
            
            # Note: Removed old dmg_reduction and the (not frail_ally) multiplier
            # because ko_threshold_value captures this more precisely

        elif st.name in ("PSN", "TOX"):
            benefit += 10.0 + (4.0 if setup_ally else 0.0)

        elif st.name in ("SLP", "FRZ"):
            benefit += 18.0 + (14.0 if setup_ally else 0.0)

        benefit *= wincon_mult

        if benefit > best:
            second = best
            best = benefit
        elif benefit > second:
            second = benefit

    return best + 0.5 * second

def _burn_chip_damage_value(battle: Any, ctx: EvalContext) -> float:
    """
    Value of passive burn damage over expected remaining turns.
    Burn deals 1/16 HP per turn.
    """
    opp = ctx.opp
    if opp is None:
        return 0.0
    
    # Estimate how many turns opponent will stay in
    # Conservative since they might switch
    our_remaining = remaining_count(battle.team)
    opp_remaining = remaining_count(battle.opponent_team)
    
    # Longer games = more chip value
    if opp_remaining >= 4 and our_remaining >= 4:
        expected_turns = 3.0
    elif opp_remaining >= 3:
        expected_turns = 2.5
    else:
        expected_turns = 2.0
    
    # 1/16 HP per turn (6.25%)
    chip_per_turn = 1.0 / 16.0
    total_chip = chip_per_turn * expected_turns
    
    # Scale to scoring system (similar to damage value)
    return total_chip * 85.0


def _status_matchup_penalty(move: Any, battle: Any, ctx: EvalContext, pressure) -> float:
    """
    Additional tempo risk for bad matchups in the CURRENT 1v1.
    
    This is applied ON TOP OF base tempo risk and evaluates:
    - Burn vs special attacker (current active pokemon is special)
    - Para vs something we outspeed (speed control less urgent NOW)
    
    Note: This is matchup-specific, NOT team-wide.
    Team considerations are handled in team_synergy_value().
    """
    st = getattr(move, "status", None)
    me = ctx.me
    opp = ctx.opp
    
    if st is None or opp is None or me is None:
        return 0.0
    
    penalty = 0.0
    
    if st == Status.BRN:
        # Current opponent is special = bad burn target
        special_prob = 1.0 - pressure.physical_prob
        penalty += special_prob * 25.0  
        
        # No immediate setup threat = less urgent
        if pressure.setup_prob < 0.2:
            penalty += 6.0
    
    if st == Status.PAR:
        # We already outspeed = less urgent NOW
        if not is_slower(me, opp):
            penalty += 10.0
        
        # No immediate setup threat = less urgent
        if pressure.setup_prob < 0.2:
            penalty += 5.0
    
    return penalty

def _poison_chip_value(battle: Any, ctx: EvalContext, is_toxic: bool) -> float:
    """
    Calculate expected chip damage over game horizon.
    
    Regular poison: 1/16 per turn (linear)
    Toxic: 1/16, 2/16, 3/16... (ramping)
    """
    our_remaining = remaining_count(battle.team)
    opp_remaining = remaining_count(battle.opponent_team)
    
    # Estimate how long opponent stays in / game continues
    if opp_remaining >= 4 and our_remaining >= 4:
        horizon = 5  # Long game
    elif opp_remaining >= 3:
        horizon = 4
    else:
        horizon = 3
    
    if is_toxic:
        # Toxic: Ramping damage
        # Turn 1: 1/16, Turn 2: 2/16, Turn 3: 3/16, etc.
        # Cumulative: 1/16 + 2/16 + 3/16 + 4/16 + 5/16 = 15/16 = 93.75% over 5 turns!
        cumulative = 0.0
        for turn in range(1, horizon + 1):
            cumulative += turn / 16.0
        
        # But opponent might switch or die early
        # Conservative estimate: assume ~60% stays in
        effective_damage = cumulative * 0.60
        
        # Scale to scoring system (similar to burn chip)
        return effective_damage * 120.0  # Higher multiplier than burn due to ramp
    
    else:
        # Regular poison: Linear
        total_chip = (1.0 / 16.0) * horizon
        return total_chip * 85.0
    
def _poison_setup_deterrence(battle: Any, ctx: EvalContext, pressure, opp: Any) -> float:
    """
    Poison deters setup sweepers by putting them on a timer.
    
    Value increases with:
    - Opponent's setup probability
    - Opponent's bulk (more turns to set up = more poison damage)
    - Lack of Rest/cleric support
    """
    setup_prob = pressure.setup_prob
    
    if setup_prob < 0.3:
        # Not a setup threat
        return 0.0
    
    value = 0.0
    
    # Base setup deterrence
    value += 25.0 * setup_prob
    
    # Check if opponent is bulky (more turns = more poison value)
    opp_hp = hp_frac(opp)
    if opp_hp > 0.80:
        value += 15.0 * setup_prob
    
    # Check if opponent has setup moves (DD, SD, NP, CM, etc.)
    try:
        dist = get_opponent_set_distribution(opp, 9) or []
        has_setup = sum(w for c, w in dist if getattr(c, "has_setup", False))
        
        if has_setup > 0.5:
            # Likely has setup move
            value += 10.0
    except Exception:
        pass
    
    return value

def _toxic_stall_synergy(battle: Any, ctx: EvalContext, opp: Any) -> float:
    """
    Massive value when you have a wall that can stall out Toxic.
    
    Toxic + Wall + Recovery = Guaranteed KO
    """
    # Check each teammate
    best_wall_value = 0.0
    
    for ally in battle.team.values():
        if ally is None or ally.fainted or ally == ctx.me:
            continue
        
        # Estimate how much damage opponent does to this ally
        # This is a simplified check - ideally we'd use full pressure calculation
        damage_to_ally = _estimate_damage_to_ally(ally, opp, battle)
        
        if damage_to_ally < 0.15:
            # Ally walls opponent!
            
            # Does ally have recovery?
            has_recovery = _has_recovery_move(ally)
            
            if has_recovery:
                # PERFECT TOXIC STALL SETUP!
                # This is a win condition
                wall_value = 50.0
                
                # Bonus if wall is very safe (<10% damage)
                if damage_to_ally < 0.10:
                    wall_value += 15.0
                
                best_wall_value = max(best_wall_value, wall_value)
            
            else:
                # Can wall but can't sustain forever
                # Still good for short-term stall
                best_wall_value = max(best_wall_value, 25.0)
    
    return best_wall_value


def _has_recovery_move(pokemon: Any) -> bool:
    """Check if Pokemon has a recovery move."""
    recovery_moves = {
        "recover", "softboiled", "roost", "rest", "slackoff",
        "moonlight", "morningsun", "synthesis", "shoreup", "wish",
        "healorder", "milkdrink", "junglehealing", "lunarblessing",
        "lifedew", "floralhealing", "strengthsap"
    }
    
    try:
        for move in pokemon.moves.values():
            if move.id in recovery_moves:
                return True
    except Exception:
        pass
    
    return False


def _get_pokemon_identifier(pokemon: Any, battle: Any) -> Optional[str]:
    """
    Get the identifier for a Pokemon object.
    
    Identifiers have format like "p1: Suicune" or "p2: Salamence".
    This is needed for poke-env's damage calculator.
    
    :param pokemon: Pokemon object to get identifier for
    :param battle: Battle object containing team dictionaries
    :return: Identifier string or None if not found
    """
    if pokemon is None:
        return None
    
    # Check player's team
    for identifier, pkmn in battle.team.items():
        if pkmn is pokemon:
            return identifier
    
    # Check opponent's team
    for identifier, pkmn in battle.opponent_team.items():
        if pkmn is pokemon:
            return identifier
    
    return None


def _estimate_damage_to_ally(ally: Any, opp: Any, battle: Any) -> float:
    """
    Estimate damage opponent deals to potential switch-in.
    
    If poke-env 0.10.0+ is available: Uses built-in damage calculator
    Otherwise: Falls back to pressure estimation
    
    Returns: Fraction of ally HP opponent deals per turn (0.0 to 2.0+)
    """
    if ally is None or opp is None:
        return 0.25
    
    # Try real damage calculator if available (poke-env 0.10.0+)
    if HAS_DAMAGE_CALC:
        try:
            # Get identifiers for both Pokemon (e.g., "p1: Suicune", "p2: Salamence")
            ally_identifier = _get_pokemon_identifier(ally, battle)
            opp_identifier = _get_pokemon_identifier(opp, battle)
            
            if ally_identifier is None or opp_identifier is None:
                # Couldn't get identifiers, fall back
                return _estimate_damage_via_pressure(ally, opp, battle)
            
            # Find opponent's best move vs this ally
            best_avg_damage = 0.0
            
            for move in opp.moves.values():
                if move is None:
                    continue
                
                # Skip status moves
                if getattr(move, "category", None) == MoveCategory.STATUS:
                    continue
                
                try:
                    # Use poke-env's ACTUAL damage calculator!
                    # Returns (min_damage, max_damage) as integers (HP lost)
                    min_dmg, max_dmg = calculate_damage(
                        attacker_identifier=opp_identifier,
                        defender_identifier=ally_identifier,
                        move=move,
                        battle=battle,
                        is_critical=False,
                    )
                    
                    # Convert to fractions of max HP
                    ally_max_hp = getattr(ally, 'max_hp', None)
                    if ally_max_hp is None or ally_max_hp <= 0:
                        ally_max_hp = getattr(ally, 'stats', {}).get('hp', 100)
                    
                    if ally_max_hp <= 0:
                        continue
                    
                    min_frac = min_dmg / ally_max_hp
                    max_frac = max_dmg / ally_max_hp
                    
                    # Average damage
                    avg_dmg = (min_frac + max_frac) / 2.0
                    
                    if avg_dmg > best_avg_damage:
                        best_avg_damage = avg_dmg
                    
                except Exception:
                    # If calc fails for this move, continue to next
                    continue
            
            if best_avg_damage > 0:
                return best_avg_damage
            else:
                # No valid moves found, fall back
                return _type_based_damage_estimate(ally, opp)
        
        except Exception:
            # Any other error with damage calc, fall back to pressure
            return _estimate_damage_via_pressure(ally, opp, battle)
    
    else:
        # No damage calculator available, use pressure estimation
        return _estimate_damage_via_pressure(ally, opp, battle)


def _estimate_damage_via_pressure(ally: Any, opp: Any, battle: Any) -> float:
    """
    Estimate damage using pressure calculation (fallback for poke-env < 0.10.0).
    
    Returns: Fraction of ally HP opponent deals per turn (0.0 to 2.0+)
    """
    try:
        # Create temporary context with ally as active Pokemon
        temp_ctx = EvalContext(
            me=ally,
            opp=opp,
            battle=battle, 
            cache={},  # Fresh cache for this calculation
        )
        
        # Run full pressure estimation
        pressure = estimate_opponent_pressure(battle, temp_ctx)
        
        # Return the damage fraction
        return pressure.damage_to_me_frac
    
    except Exception:
        # Ultimate fallback to type-based heuristic
        return _type_based_damage_estimate(ally, opp)


def _type_based_damage_estimate(ally: Any, opp: Any) -> float:
    """
    Fallback damage estimate based on type matchups.
    Used when full pressure calculation fails.
    """
    try:
        opp_types = safe_types(opp)
        ally_types = safe_types(ally)
        
        # Start with average
        estimate = 0.25
        
        # Check for common walls
        if PokemonType.STEEL in ally_types:
            # Steel resists many types
            estimate = 0.12
            
            # But weak to Fighting/Fire/Ground
            if PokemonType.FIGHTING in opp_types:
                estimate = 0.40
            elif PokemonType.FIRE in opp_types:
                estimate = 0.45
            elif PokemonType.GROUND in opp_types:
                estimate = 0.40
        
        elif PokemonType.FAIRY in ally_types:
            # Fairy resists Fighting/Bug/Dark, immune to Dragon
            estimate = 0.18
            
            if PokemonType.DRAGON in opp_types:
                estimate = 0.05  # Immune to Dragon
            elif PokemonType.STEEL in opp_types or PokemonType.POISON in opp_types:
                estimate = 0.40  # Weak to Steel/Poison
        
        elif PokemonType.WATER in ally_types:
            estimate = 0.22
            
            if PokemonType.ELECTRIC in opp_types or PokemonType.GRASS in opp_types:
                estimate = 0.45
        
        # More type checks could be added here
        
        return min(2.0, max(0.05, estimate))
    
    except Exception:
        # Ultimate fallback
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
    
    Key insight: What matters is HITS TO KO (HTK), not raw damage.
    
    Returns:
    - Large bonus if burn saves from OHKO/2HKO (+35 to +65)
    - Small bonus if burn improves survivability meaningfully (+10 to +28)
    - Penalty if burn doesn't change HTK enough - wasted on tanky matchup (-15)
    
    Examples:
    - Weavile takes 75% per turn → 2HKO, with burn → 3HKO: +35 * phys_p
    - Corviknight takes 20% per turn → 5HKO, with burn → 10HKO: -15 * phys_p (wasted!)
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
    print("extra hits: " + str(extra_hits))
    
    # Scale by physical probability
    value = 0.0
    
    # === CRITICAL THRESHOLDS ===
    
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

    
def _poison_counterplay_penalty(battle: Any, ctx: EvalContext, opp: Any) -> float:
    """
    Reduce poison value if opponent can cure it.
    """
    penalty = 0.0
    
    # Check opponent's known moves for Rest
    try:
        for move in opp.moves.values():
            if move.id == "rest":
                # Opponent has Rest
                # They can cure poison, but have to sleep
                # This is still somewhat good for us (they lose 2 turns)
                penalty += 20.0
                break
    except Exception:
        pass
    
    # Check opponent's ability
    try:
        ability = opp.ability
        if ability in ("magicguard", "magicbounce"):
            # Immune to poison damage or bounces it back
            penalty += 100.0  # Don't use poison!
    except Exception:
        pass
    
    # Check for cleric support on opponent's team
    # This is complex - requires checking all opponent's team
    # For now, skip (low priority)
    
    return penalty

def _poison_total_value(battle: Any, ctx: EvalContext, pressure, move) -> float:
    """
    Comprehensive poison/toxic value.
    
    NEW: Accounts for:
    - Ramping damage (Toxic)
    - Setup deterrence
    - Toxic stall synergy
    - Opponent counterplay
    """
    opp = ctx.opp
    if opp is None:
        return 0.0
    
    move_id = str(getattr(move, "id", "")).lower()
    is_toxic = (move_id == "toxic")
    
    # Base value
    value = BASE_STATUS_POISON  # 32.0
    
    # 1. Chip damage (ramping for Toxic)
    chip_value = _poison_chip_value(battle, ctx, is_toxic)
    value += chip_value
    
    # 2. Setup deterrence (puts sweepers on timer)
    setup_value = _poison_setup_deterrence(battle, ctx, pressure, opp)
    value += setup_value
    
    # 3. Toxic stall synergy (wall + recovery)
    if is_toxic:
        stall_value = _toxic_stall_synergy(battle, ctx, opp)
        value += stall_value
    
    # 4. Opponent counterplay (Rest, immunities)
    counterplay = _poison_counterplay_penalty(battle, ctx, opp)
    value -= counterplay
    
    return value