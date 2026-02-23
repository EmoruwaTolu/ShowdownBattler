from typing import Any
import math

from poke_env.battle import MoveCategory

from bot.model.ctx import EvalContext
from bot.scoring.damage_score import (
    estimate_damage_fraction,
    ko_probability_from_fraction,
)
from bot.scoring.helpers import hp_frac, is_slower
from bot.scoring.opponent_pressure import estimate_opponent_pressure
from bot.scoring.status_score import score_status_move
from bot.scoring.secondary_score import score_secondaries
from poke_env.battle import MoveCategory, SideCondition, Status
from bot.mcts.shadow_state import get_move_boosts

# Canonical set of moves that apply the lockedmove volatile status.
# These lock the user in for 2-3 turns with no choice; confusion at the end.
_LOCK_MOVE_IDS: frozenset = frozenset({"outrage", "thrash", "petaldance", "ragingfury"})


def _is_lock_move(move: Any) -> bool:
    """
    True if this move locks the user in for multiple turns (Outrage, Thrash, etc.)

    Uses move ID as the primary signal — it's always reliably set on both real
    and mock Move objects. volatile_status is checked as a secondary fallback
    to catch any moves not in the explicit list.
    """
    move_id = str(getattr(move, "id", "") or "").lower().replace(" ", "").replace("-", "")
    if move_id in _LOCK_MOVE_IDS:
        return True
    # Fallback: check volatile_status attribute and raw entry data
    volatile = getattr(move, "volatile_status", None)
    if volatile is not None and "locked" in str(volatile).lower():
        return True
    try:
        entry = getattr(move, "entry", {}) or {}
        self_vs = str((entry.get("self", {}) or {}).get("volatileStatus", "")).lower()
        if "lockedmove" in self_vs:
            return True
    except Exception:
        pass
    return False


def _lock_immune_penalty(move: Any, me: Any, opp: Any, battle: Any, ko_prob: float) -> float:
    """
    Additional penalty for lock-in moves (Outrage, Thrash, etc.) when the opponent
    has alive bench members that are immune or strongly resist the move type.

    The core problem: if we're locked into Outrage and Clefable is on the bench,
    the opponent can freely pivot to Clefable, waste our turns, and punish freely.

    Scaled by:
    - bench_hp: a healthier immune pivot is a safer/more dangerous switch-in
    - (1 - ko_prob): if we're about to KO the active, they can't switch in time
    - being slower: opponent can predict-switch on the current turn itself
    """
    if not _is_lock_move(move):
        return 0.0
    if ko_prob >= 0.90:
        return 0.0  # near-guaranteed KO: lock ends when they faint

    opp_team = getattr(battle, "opponent_team", {}) or {}
    penalty = 0.0

    for opp_mon in opp_team.values():
        if opp_mon is opp:
            continue  # skip the active opponent
        bench_hp = float(getattr(opp_mon, "current_hp_fraction", 0) or 0)
        if bench_hp <= 0:
            continue

        try:
            dmg = float(estimate_damage_fraction(move, me, opp_mon, battle))
        except Exception:
            continue  # can't compute: don't assume immunity

        if dmg < 0.01:
            # Immune pivot: free switch-in, we waste 2+ turns doing nothing
            penalty += 22.0 * bench_hp * (1.0 - ko_prob)
        elif dmg < 0.025:
            # Strong resist (~0.25x): still a highly favourable switch-in for them
            penalty += 7.0 * bench_hp * (1.0 - ko_prob)

    # Being slower means opponent can prediction-switch on the current turn
    if penalty > 0 and is_slower(me, opp):
        penalty *= 1.25

    return min(penalty, 40.0)


def _opp_priority_ko_threat(opp: Any, me: Any, battle: Any) -> bool:
    """True if opponent has a known damaging priority move that can KO our active mon."""
    my_hp = hp_frac(me)
    if my_hp <= 0.0:
        return False
    try:
        for mv in (getattr(opp, 'moves', {}) or {}).values():
            if mv is None:
                continue
            if int(getattr(mv, 'priority', 0) or 0) > 0 and int(getattr(mv, 'base_power', 0) or 0) > 0:
                try:
                    if float(estimate_damage_fraction(mv, opp, me, battle)) >= my_hp:
                        return True
                except Exception:
                    pass
    except Exception:
        pass
    return False


def score_move(move: Any, battle: Any, ctx: EvalContext) -> float:
    me = ctx.me
    opp = ctx.opp
    if me is None or opp is None:
        return -100.0

    # Setup moves (Dragon Dance / Calm Mind etc.)
    setup_score = score_setup_move(move, battle, ctx)
    if setup_score > 0:
        return setup_score

    # Other status moves
    if move.category == MoveCategory.STATUS:
        return score_status_move(move, battle, ctx)

    dmg_frac = float(estimate_damage_fraction(move, me, opp, battle))
    opp_hp = hp_frac(opp)

    accuracy = float(getattr(move, "accuracy", 1.0) or 1.0)
    accuracy = max(0.0, min(1.0, accuracy))

    # Expected damage
    score = (dmg_frac * 100.0) * accuracy

    # Small "reliability" bonus — capped at +2 to avoid over-rewarding 100% accuracy
    # on top of already-accurate expected-damage math
    score += min(2.0, 5.0 * (accuracy - 0.85) / 0.15) if accuracy >= 0.85 else -10.0

    # KO Bonus
    ko_prob = ko_probability_from_fraction(dmg_frac, opp_hp)
    if ko_prob > 0:
        slower = is_slower(me, opp)

        # Finishing is valuable, but keep it proportional and not bigger than damage itself
        # If you're faster, KO is slightly more valuable (avoid taking a hit)
        finish_bonus = (30.0 + (10.0 if not slower else 0.0)) * ko_prob
        score += finish_bonus

    if ko_prob < 0.95:
        score += score_secondaries(move, battle, ctx, ko_prob, dmg_frac=dmg_frac)

    priority = int(getattr(move, "priority", 0) or 0)
    if priority > 0:
        # priority matters most when you're slower OR opp is low
        if opp_hp < 0.35:
            score += 10.0
        elif is_slower(me, opp):
            score += 6.0
        else:
            score += 2.0

    score -= get_stat_drop_penalty(move, battle, ctx)

    recoil = float(getattr(move, "recoil", 0) or 0.0)
    if recoil > 0:
        recoil_penalty = min(20.0, abs(recoil) * 50.0)
        score -= recoil_penalty

    # Recharge penalty: moves like Hyper Beam/Blast Burn force a wasted turn after use.
    # KO prob scales it down — if they're fainting, the lost turn doesn't matter.
    flags = getattr(move, "flags", set()) or set()
    has_recharge = "recharge" in flags or bool(getattr(move, "recharge", False))
    if has_recharge:
        score -= 15.0 * max(0.0, 1.0 - ko_prob)

    # Self-lock penalty: moves like Outrage/Thrash lock you in for 2-3 turns,
    # removing all switching/pivoting options until the lock breaks.
    if _is_lock_move(move):
        score -= 10.0 * max(0.0, 1.0 - ko_prob)

    # Immune-pivot penalty: if the opponent has an alive bench member that is
    # immune (or strongly resists) the lock move, they can freely pivot to it
    # and wall us for the remainder of the lock duration.
    score -= _lock_immune_penalty(move, me, opp, battle, ko_prob)

    # Crit is probably affecting stability, will soon change to not boost moves with regular crit chance jusr moves with a heightened one
    score += min(3.0, calculate_crit_bonus(move, battle, ctx, dmg_frac, ko_prob))

    # Priority KO threat: if opponent likely has a priority move that KOs us this turn
    # and our move has no priority, we'll be dead before this move fires.
    move_priority = int(getattr(move, "priority", 0) or 0)
    if move_priority <= 0:
        if _opp_priority_ko_threat(opp, me, battle):
            # Known revealed KO threat — heavy but not absolute
            # (could be choice-locked, low-likelihood to click, etc.)
            score *= 0.20
        else:
            # Belief-based: unrevealed priority move may exist; scale by how low our HP is.
            # Smooth linear factor avoids hard oscillation when HP hovers near a threshold.
            # hp_factor: 0.0 at 67%+ HP, grows to 1.0 at 0% HP.
            my_hp_val = hp_frac(me)
            try:
                pressure = estimate_opponent_pressure(battle, ctx)
                hp_factor = max(0.0, 1.0 - my_hp_val * 1.5)
                p_threat = min(1.0, pressure.priority_p) * hp_factor
                if p_threat > 0.05:
                    score *= (1.0 - 0.6 * p_threat)
            except Exception:
                pass

    return float(score)

def calculate_crit_bonus(move: Any, battle: Any, ctx: Any, base_damage_frac: float, ko_prob: float) -> float:
    """
    Calculate bonus value for critical hit chance.
    
    Args:
        move: The move being evaluated
        battle: Battle state
        ctx: Evaluation context
        base_damage_frac: Non-crit damage as fraction (from estimate_damage_fraction)
        ko_prob: KO probability without crit
        
    Returns:
        Bonus points (0-40)
    """
    me = ctx.me
    opp = ctx.opp
    
    if me is None or opp is None:
        return 0.0
    
    if move.category == MoveCategory.STATUS:
        return 0.0
    
    # Get crit rate — only apply bonus for truly high-crit moves (ratio >= 2)
    # ratio 0 (4.17%) and ratio 1 / Focus Energy (12.5%) are too low to be meaningful
    # and create "coinflip-chasing" instability in close lines
    crit_ratio = getattr(move, 'crit_ratio', 0) or 0
    if crit_ratio < 2:
        return 0.0
    crit_chance = get_crit_chance(crit_ratio)
    
    bonus = 0.0
    
    boost_value = calculate_boost_ignore_value(
        move, me, opp, battle, crit_chance
    )
    bonus += boost_value
    
    # Only consider if we don't already guaranteed KO
    if ko_prob < 0.90:
        htk_value = _calculate_htk_improvement_value(
            base_damage_frac, crit_chance, ctx
        )
        bonus += htk_value
    
    return bonus


def get_crit_chance(crit_ratio: int) -> float:
    """
    Convert crit ratio to actual probability.
    
    Gen 9 crit rates:
    - ratio 0 (normal): 1/24 = 4.17%
    - ratio 1 (Focus Energy): 1/8 = 12.5%
    - ratio 2 (Stone Edge, Razor Leaf): 1/2 = 50%
    - ratio 3+ (Frost Breath): 100%
    """
    if crit_ratio <= 0:
        return 1.0 / 24.0  # ~4.17%
    elif crit_ratio == 1:
        return 1.0 / 8.0   # 12.5%
    elif crit_ratio == 2:
        return 0.5         # 50%
    else:
        return 1.0         # 100%

def get_stat_drop_penalty(move: Any, battle: Any, ctx: EvalContext) -> float:
    """
    Penalty for moves that drop our own stats.
    
    Returns positive value to subtract from score.
    """
    
    boost_data = get_move_boosts(move)
    if not boost_data:
        return 0.0
    
    self_boosts, target_boosts, chance = boost_data
    if not self_boosts:
        return 0.0
    
    # Calculate penalty
    penalty = 0.0
    for stat, stages in self_boosts.items():
        if stages < 0:  # Only penalize drops
            if stat in ['atk', 'spa']:
                penalty += abs(stages) * 15.0  # -2 SpA = -30 points
            elif stat == 'spe':
                penalty += abs(stages) * 10.0
            else:
                penalty += abs(stages) * 5.0
    
    # Reduce penalty if opponent almost dead
    opp_hp = getattr(ctx.opp, 'current_hp_fraction', 1.0)
    if opp_hp < 0.3:
        penalty *= 0.5  # Worth it for the KO
    
    # Increase penalty in sweep scenarios
    my_hp = getattr(ctx.me, 'current_hp_fraction', 1.0)
    if my_hp > 0.7 and opp_hp > 0.5:
        penalty *= 1.3  # Need sustained damage
    
    return penalty

def score_setup_move(move: Any, battle: Any, ctx: EvalContext) -> float:
    """
    Score stat-boosting moves (Swords Dance, Nasty Plot, Dragon Dance, etc.)

    Key ideas:
    - Diminishing returns per stage (strong early, weak late)
    - Risk-aware: setup is only good if we likely survive
    - Speed boosts are only valuable if they flip speed order
    - Depth=3 horizon: repeated setup beyond +1/+2 should be strongly discouraged
    - Cap output so priors don't dominate everything
    """

    boost_data = get_move_boosts(move)
    if not boost_data:
        return 0.0

    self_boosts, target_boosts, chance = boost_data

    # We only care about self-boosts for setup moves
    if not self_boosts:
        return 0.0

    # Only positive boosts (setup moves, not Draco Meteor penalties)
    if all(v <= 0 for v in self_boosts.values()):
        return 0.0

    me = ctx.me
    opp = ctx.opp
    current_boosts = getattr(me, "boosts", {}) or {}

    # Base value with diminishing returns
    boost_value = 0.0

    for stat, stages in self_boosts.items():
        if stages <= 0:
            continue

        current_level = int(current_boosts.get(stat, 0))
        if current_level >= 6:
            continue

        actual_stages = min(int(stages), 6 - current_level)

        # base value per stage: keep these reasonable
        base_per_stage = 30.0 if stat in ["atk", "spa"] else 20.0 if stat == "spe" else 12.0

        for i in range(actual_stages):
            new_level = current_level + i + 1

            if new_level <= 2:
                multiplier = 1.0
            elif new_level == 3:
                multiplier = 0.7
            elif new_level == 4:
                multiplier = 0.5
            elif new_level == 5:
                multiplier = 0.3
            else:  # 6
                multiplier = 0.1

            boost_value += base_per_stage * multiplier

    if boost_value <= 0.0:
        return 0.0

    # Risk: can we survive the turn we spend setting up?
    my_hp = float(getattr(me, "current_hp_fraction", 1.0) or 1.0)
    opp_hp = float(getattr(opp, "current_hp_fraction", 1.0) or 1.0)

    opp_max_damage = 0.0
    _found_any_dmg = False
    for opp_move in getattr(opp, "moves", {}).values():
        try:
            dmg = float(estimate_damage_fraction(opp_move, opp, me, battle))
            opp_max_damage = max(opp_max_damage, dmg)
            _found_any_dmg = True
        except Exception:
            pass  # skip; don't inject per-move fallback
    if not _found_any_dmg:
        # No moves calculated at all — use a modest estimate rather than 0.5
        opp_max_damage = 0.35

    # risk scaling (slightly harsher than before)
    if opp_max_damage >= my_hp:
        boost_value *= 0.15
    elif opp_max_damage >= my_hp * 0.75:
        boost_value *= 0.35
    elif opp_max_damage >= my_hp * 0.5:
        boost_value *= 0.55
    else:
        boost_value *= 1.10

    # 2HKO hard gate: if opponent can KO us in ≤2 hits and setup doesn't include
    # a speed boost (which could let us move first and avoid the second hit),
    # setup is almost always a misplay — we'll be KO'd before the boosts pay off.
    if opp_max_damage > 0:
        setup_has_speed = self_boosts.get('spe', 0) > 0
        htk = math.ceil(my_hp / opp_max_damage)
        if htk <= 2 and not setup_has_speed:
            boost_value *= 0.30

    # Belief-based priority gate: the revealed-move scan above misses unrevealed
    # priority moves. If belief says opponent likely has priority and we're low HP,
    # setup is nearly pointless — we'll be KO'd before we can use the boosts.
    # Smooth hp_factor avoids hard oscillation near a threshold.
    try:
        pressure = estimate_opponent_pressure(battle, ctx)
        hp_factor = max(0.0, 1.0 - my_hp * 1.5)  # 0.0 at 67%+ HP, 1.0 at 0%
        p_prio_ko = min(1.0, pressure.priority_p) * hp_factor
        if p_prio_ko > 0.1:
            boost_value *= max(0.10, 1.0 - 0.85 * p_prio_ko)
    except Exception:
        pass

    # Speed: reward only if it flips speed order
    # Account for paralysis, Choice Scarf, and Tailwind
    try:
        def _eff_spe_base(pokemon: Any) -> float:
            """Base effective speed: stat × status × item × side condition."""
            spe = float((pokemon.stats or {}).get("spe", 100) or 100)
            if getattr(pokemon, "status", None) == Status.PAR:
                spe *= 0.5
            item = str(getattr(pokemon, "item", "") or "").lower().replace(" ", "").replace("-", "")
            if item == "choicescarf":
                spe *= 1.5
            return spe

        my_spe = _eff_spe_base(me)
        opp_spe = _eff_spe_base(opp)

        # Tailwind doubles speed for that side
        try:
            my_side = getattr(battle, "side_conditions", {}) or {}
            opp_side = getattr(battle, "opponent_side_conditions", {}) or {}
            if SideCondition.TAILWIND in my_side:
                my_spe *= 2.0
            if SideCondition.TAILWIND in opp_side:
                opp_spe *= 2.0
        except Exception:
            pass

        cur_spe_stage = int(current_boosts.get("spe", 0))
        gained_spe = int(self_boosts.get("spe", 0))

        def spe_multiplier(stage: int) -> float:
            stage = max(-6, min(6, stage))
            return (2.0 + stage) / 2.0 if stage >= 0 else 2.0 / (2.0 - stage)

        before = my_spe * spe_multiplier(cur_spe_stage)
        after = my_spe * spe_multiplier(cur_spe_stage + gained_spe)

        was_slower = before < opp_spe
        becomes_faster = after >= opp_spe

        if was_slower and becomes_faster:
            boost_value *= 1.20  # big value: you now move first
        elif was_slower and not becomes_faster and gained_spe > 0:
            boost_value *= 0.75  # still slower: meh
        else:
            boost_value *= 0.95  # already faster or no speed boost: small change
    except Exception:
        pass

    # Multi-stat boost bonus
    num_boosted_stats = sum(1 for v in self_boosts.values() if v > 0)
    if num_boosted_stats >= 2:
        boost_value *= 1.10 

    # HP situation
    if my_hp > 0.8 and opp_hp > 0.6:
        boost_value *= 1.05 
    elif opp_hp < 0.3:
        boost_value *= 0.45

    # Horizon factor for depth=3: discourage repeated setup
    atk_stage = int(current_boosts.get("atk", 0))
    spa_stage = int(current_boosts.get("spa", 0))
    spe_stage = int(current_boosts.get("spe", 0))

    max_stage = max(atk_stage, spa_stage, spe_stage)

    if max_stage >= 4:
        boost_value *= 0.50  # Only penalize extreme setup
    elif max_stage >= 2:
        boost_value *= 0.80  # Mild penalty
    elif max_stage >= 1:
        boost_value *= 0.90  # Very mild penalty

    # Final cap to prevent prior domination
    boost_value = min(boost_value, 70.0)

    # print("Boost value: " + str(boost_value))

    return float(boost_value)

def calculate_boost_ignore_value(move: Any, me: Any, opp: Any, battle: Any, crit_chance: float) -> float:
    """
    Calculate value of ignoring stat boosts/drops via crit.
    
    Crits ignore:
    - Opponent's positive defensive boosts
    - Our negative offensive boosts  
    - Screens (Reflect/Light Screen)
    
    Returns:
        Bonus points (0-25)
    """
    value = 0.0
    
    # Determine which stats matter for this move
    is_physical = move.category == MoveCategory.PHYSICAL
    
    try:
        opp_boosts = getattr(opp, 'boosts', {})
        defensive_stat = 'def' if is_physical else 'spd'
        opp_def_boost = opp_boosts.get(defensive_stat, 0)
        
        if opp_def_boost > 0:
            # Opponent is boosted defensively
            # Crit ignores defensive boosts
            
            # Value scales with boost level
            # +1 = moderate value, +2 = high value, +3+ = huge value
            boost_impact = {
                1: 8.0,   # +1 Def: crit is ~1.5x better
                2: 15.0,  # +2 Def: crit is ~2x better
                3: 22.0,  # +3 Def: crit is ~2.5x better
            }.get(min(opp_def_boost, 3), 22.0)
            
            # Scale by crit chance
            # 50% crit (Stone Edge) gets full value
            # 12% crit (Focus Energy) gets partial value
            value += boost_impact * crit_chance
    
    except Exception:
        pass
    
    try:
        my_boosts = getattr(me, 'boosts', {})
        offensive_stat = 'atk' if is_physical else 'spa'
        my_atk_boost = my_boosts.get(offensive_stat, 0)
        
        if my_atk_boost < 0:
            # We have attack drops (Intimidate, etc.)
            
            debuff_impact = {
                -1: 8.0,   # -1 Atk
                -2: 15.0,  # -2 Atk
                -3: 22.0,  # -3 Atk
            }.get(max(my_atk_boost, -3), 22.0)
            
            value += debuff_impact * crit_chance
    
    except Exception:
        pass
    
    try:
        # Check opponent's side conditions
        opp_side = battle.opponent_side_conditions if hasattr(battle, 'opponent_side_conditions') else {}
        
        # Reflect blocks physical, Light Screen blocks special
        screen_up = (
            (is_physical and SideCondition.REFLECT in opp_side) or
            (not is_physical and SideCondition.LIGHT_SCREEN in opp_side) or
            (SideCondition.AURORA_VEIL in opp_side)
        )
        
        if screen_up:
            # Screens halve damage, crits ignore them
            # This is HUGE value for high-crit moves
            value += 12.0 * crit_chance
    
    except Exception:
        pass
    
    return min(25.0, value)  # Cap at 25 points


def _calculate_htk_improvement_value(base_damage_frac: float, crit_chance: float, ctx: Any) -> float:
    """
    Calculate value of crit improving hits-to-KO.
    
    Key insight: Crits matter most when they change the plan!
    
    Examples:
    - 3HKO → 2HKO with crit: HUGE value (saves a turn)
    - 2HKO → 2HKO with crit: Small value (no plan change)
    - 5HKO → 4HKO with crit: Moderate value (marginal)
    
    Returns:
        Bonus points (0-20)
    """
    if base_damage_frac <= 0:
        return 0.0
    
    opp = ctx.opp
    opp_hp = getattr(opp, 'current_hp_fraction', 1.0) or 1.0
    
    # Calculate hits to KO without crit
    htk_no_crit = math.ceil(opp_hp / max(0.01, base_damage_frac))
    
    # Calculate hits to KO with crit (1.5x damage)
    crit_damage = base_damage_frac * 1.5
    htk_with_crit = math.ceil(opp_hp / max(0.01, crit_damage))
    
    # How many turns does crit save?
    turns_saved = htk_no_crit - htk_with_crit
    
    if turns_saved <= 0:
        # Crit doesn't change the plan
        return 0.0
    
    # Value based on what changes
    if htk_no_crit >= 4 and htk_with_crit <= 2:
        # Slow KO → Fast KO (4+ turns → 2 turns)
        # This is GAME-CHANGING
        base_value = 20.0
    
    elif htk_no_crit == 3 and htk_with_crit == 2:
        # 3HKO → 2HKO
        # Very valuable (saves a turn in race)
        base_value = 15.0
    
    elif htk_no_crit == 2 and htk_with_crit == 1:
        # 2HKO → OHKO
        # Extremely valuable (no retaliation)
        base_value = 18.0
    
    elif turns_saved == 1:
        # Any other 1-turn save
        base_value = 10.0
    
    else:
        # Saves 2+ turns
        base_value = 12.0
    
    # Scale by crit chance
    # 50% crit = expect this half the time
    # 12% crit = expect this rarely
    return base_value * crit_chance