from typing import Any

from poke_env.battle import MoveCategory

from ShowdownBattler.bot.scoring.switch_score import pivot_move_bonus
from bot.model.ctx import EvalContext
from bot.scoring.damage_score import (
    estimate_damage_fraction,
    ko_probability_from_fraction,
    ko_bonus,
)
from bot.scoring.helpers import hp_frac, is_slower, remaining_count
from bot.scoring.race import evaluate_race_for_move
from bot.scoring.status_score import score_status_move
from bot.scoring.secondary_score import score_secondaries
from bot.scoring.chip_score import chip_synergy_value

from bot.scoring.pp_management import pp_conservation_penalty
from bot.scoring.pressure import estimate_opponent_pressure


def _recoil_penalty(move: Any, battle: Any, ctx: EvalContext) -> float:
    """
    Penalize recoil moves based on role preservation and remaining team depth.
    """
    me = ctx.me
    if me is None:
        return 0.0
    
    # Check if move has recoil (moves like Double-Edge, Brave Bird, Flare Blitz)
    recoil = getattr(move, "recoil", None)
    if recoil is None or (isinstance(recoil, (int, float)) and recoil <= 0):
        return 0.0
    
    my_hp = hp_frac(me)
    
    # Estimate recoil damage
    # recoil format: [numerator, denominator] or a fraction
    if isinstance(recoil, (list, tuple)) and len(recoil) >= 2:
        recoil_frac = float(recoil[0]) / float(recoil[1])
    else:
        recoil_frac = 0.25  # Default assumption (1/4 recoil)
    
    # Estimate damage we'll deal to calculate actual recoil
    dmg_dealt = estimate_damage_fraction(move, me, ctx.opp, battle) if ctx.opp else 0.3
    estimated_recoil = dmg_dealt * recoil_frac
    
    # Calculate HP after recoil
    hp_after = my_hp - estimated_recoil
    
    # Get opponent pressure to see if we need to survive for role preservation
    pressure = estimate_opponent_pressure(battle, ctx)
    
    # Penalty factors:
    penalty = 0.0
    
    # 1. Team depth penalty - if we're down in Pokemon count, preserve HP
    our_remaining = remaining_count(battle.team)
    opp_remaining = remaining_count(battle.opponent_team)
    
    if our_remaining < opp_remaining:
        depth_penalty = (opp_remaining - our_remaining) * 8.0
        penalty += depth_penalty
    
    # 2. Role preservation - if recoil puts us in revenge kill range
    if hp_after < pressure.damage_to_me_frac * 1.1:
        # We'd be in OHKO range after recoil
        penalty += 25.0
    elif hp_after < pressure.damage_to_me_frac * 1.5:
        # We'd be in 2HKO range after recoil (still risky)
        penalty += 12.0
    
    # 3. Critical HP thresholds
    if my_hp > 0.5 and hp_after <= 0.5:
        # Crossing 50% threshold (common bulk breakpoint)
        penalty += 12.0
    
    if hp_after < 0.25:
        # Very low HP after recoil - limits future utility
        penalty += 18.0
    
    # 4. If we're already low, recoil is extra bad
    if my_hp < 0.35:
        penalty += 15.0
    
    # 5. Scale by how much we need this Pokemon later
    # If opponent has setup threats and we're a check, preserve HP
    if pressure.setup_prob > 0.4:
        penalty += 10.0 * pressure.setup_prob
    
    # 6. If opponent has priority threats, being at low HP is extra dangerous
    if pressure.priority_prob > 0.3 and hp_after < 0.4:
        penalty += 8.0 * pressure.priority_prob
    
    return penalty


def _self_destruct_penalty(move: Any, battle: Any, ctx: EvalContext, ko_prob: float) -> float:
    """
    Heavy penalty for self-destruct/explosion moves.
    
    Penalty is reduced when:
    - We're getting a guaranteed KO (high ko_prob)
    - Opponent is a major threat to our team (high pressure)
    - We're low HP anyway (sacrifice is more acceptable)
    - We're behind in Pokemon count
    """
    move_id = str(getattr(move, "id", "")).lower()
    
    if move_id not in ("selfdestruct", "explosion", "mistyexplosion", "healingwish", "lunardance"):
        return 0.0
    
    # Base penalty for sacrificing ourselves
    penalty = 60.0
    
    our_remaining = remaining_count(battle.team)
    opp_remaining = remaining_count(battle.opponent_team)
    
    # If we're ahead in Pokemon, self-destructing is worse
    if our_remaining > opp_remaining:
        penalty += (our_remaining - opp_remaining) * 20.0
    
    # If we're behind, it's slightly less bad
    if our_remaining < opp_remaining:
        penalty -= (opp_remaining - our_remaining) * 8.0
    
    # If we're low HP anyway, sacrifice is more acceptable
    me = ctx.me
    if me is not None:
        my_hp = hp_frac(me)
        if my_hp < 0.3:
            penalty -= 20.0
        elif my_hp < 0.5:
            penalty -= 10.0
    
    # Reduce penalty for guaranteed KOs
    if ko_prob >= 0.90:
        # Very likely KO - much more acceptable
        penalty -= 35.0
    elif ko_prob >= 0.70:
        # Probable KO - somewhat acceptable
        penalty -= 20.0
    elif ko_prob >= 0.50:
        # Coin flip KO - slightly acceptable
        penalty -= 10.0
    
    # Reduce penalty based on how threatening opponent is to our team
    pressure = estimate_opponent_pressure(battle, ctx)
    
    # High threat opponents are worth trading for
    threat_value = 0.0
    
    # Setup sweepers are dangerous
    if pressure.setup_prob > 0.4:
        threat_value += 15.0 * pressure.setup_prob
    
    # High damage threats
    if pressure.damage_to_me_frac > 0.35:
        threat_value += 20.0 * (pressure.damage_to_me_frac - 0.35)
    
    # Priority threats to frail teammates
    if pressure.priority_prob > 0.3:
        threat_value += 10.0 * pressure.priority_prob
    
    penalty -= threat_value
    
    # Never let penalty go below 0 (don't actively encourage self-destruct)
    return max(0.0, penalty)


def _drain_move_bonus(move: Any, battle: Any, ctx: EvalContext) -> float:
    """
    Bonus for drain moves (Giga Drain, Drain Punch, Leech Life, etc.)
    Value increases when we're damaged and when we can survive to heal.
    
    NOTE: You still heal even if you KO the opponent, so no ko_prob scaling.
    """
    drain = getattr(move, "drain", None)
    if drain is None or (isinstance(drain, (int, float)) and drain <= 0):
        return 0.0
    
    me = ctx.me
    if me is None:
        return 0.0
    
    # Drain format: [numerator, denominator]
    if isinstance(drain, (list, tuple)) and len(drain) >= 2:
        drain_frac = float(drain[0]) / float(drain[1])
    else:
        drain_frac = 0.5  # Default (50% drain)
    
    # Estimate damage we'll deal
    dmg_dealt = estimate_damage_fraction(move, me, ctx.opp, battle) if ctx.opp else 0.25
    
    # Healing = drain_frac * damage dealt
    healing = dmg_dealt * drain_frac
    
    my_hp = hp_frac(me)
    
    # Value healing more when we're damaged
    if my_hp < 0.5:
        multiplier = 2.0
    elif my_hp < 0.7:
        multiplier = 1.5
    else:
        multiplier = 1.0
    
    # Base healing value
    bonus = healing * 50.0 * multiplier
    
    # Extra value if this healing might let us survive another hit
    pressure = estimate_opponent_pressure(battle, ctx)
    hp_after_heal = min(1.0, my_hp + healing)
    
    if my_hp < pressure.damage_to_me_frac and hp_after_heal >= pressure.damage_to_me_frac:
        # Healing takes us out of OHKO range
        bonus += 15.0
    
    return bonus

_ROCK_VS_TYPE = {
    "normal": 1.0, "fire": 2.0, "water": 1.0, "electric": 1.0, "grass": 1.0,
    "ice": 2.0, "fighting": 0.5, "poison": 1.0, "ground": 0.5, "flying": 2.0,
    "psychic": 1.0, "bug": 2.0, "rock": 1.0, "ghost": 1.0, "dragon": 1.0,
    "dark": 1.0, "steel": 0.5, "fairy": 1.0,
}

def _type_name(t: Any) -> str:
    return str(t).split(".")[-1].strip().lower()

def _has_boots(mon: Any) -> bool:
    item = str(getattr(mon, "item", "") or "").lower()
    return item in ("heavydutyboots", "heavy-duty boots", "heavy_duty_boots")

def _is_grounded(mon: Any) -> bool:
    # Simple grounded check (ignore Levitate for now; you can add later)
    try:
        types = getattr(mon, "types", None) or []
        return not any(_type_name(t) == "flying" for t in types)
    except Exception:
        return True

def _is_poison(mon: Any) -> bool:
    try:
        types = getattr(mon, "types", None) or []
        return any(_type_name(t) == "poison" for t in types)
    except Exception:
        return False

def _is_steel(mon: Any) -> bool:
    try:
        types = getattr(mon, "types", None) or []
        return any(_type_name(t) == "steel" for t in types)
    except Exception:
        return False

def _rock_multiplier(mon: Any) -> float:
    """Stealth Rock multiplier based on mon's types."""
    try:
        types = getattr(mon, "types", None) or []
        mult = 1.0
        for t in types:
            mult *= float(_ROCK_VS_TYPE.get(_type_name(t), 1.0))
        return mult
    except Exception:
        return 1.0

def _side_conds_for(mon: Any, battle: Any) -> dict:
    """
    Return side conditions dict for the side `mon` is on.
    """
    if battle is None or mon is None:
        return {}
    try:
        if mon in battle.team.values():
            return getattr(battle, "side_conditions", None) or {}
        return getattr(battle, "opponent_side_conditions", None) or {}
    except Exception:
        return {}

def _hazard_entry_damage_frac(mon: Any, battle: Any) -> float:
    """
    Approx entry damage fraction from SR + Spikes (no Toxic Spikes here).
    Boots => 0.
    """
    if mon is None or battle is None:
        return 0.0
    if _has_boots(mon):
        return 0.0

    conds = _side_conds_for(mon, battle)
    sr = int(conds.get("stealthrock", 0) or 0)
    spikes = int(conds.get("spikes", 0) or 0)

    dmg = 0.0

    # Stealth Rock: 1/8 * rock effectiveness
    if sr > 0:
        dmg += (1.0 / 8.0) * _rock_multiplier(mon)

    # Spikes: grounded only; 1 layer 1/8, 2 layers 1/6, 3 layers 1/4
    if spikes > 0 and _is_grounded(mon):
        if spikes == 1:
            dmg += 1.0 / 8.0
        elif spikes == 2:
            dmg += 1.0 / 6.0
        else:
            dmg += 1.0 / 4.0

    return float(max(0.0, dmg))

def _toxic_spikes_burden(mon: Any, battle: Any) -> float:
    """
    Soft penalty for Toxic Spikes (status pressure), not direct damage.
    Poison types absorb; Steel types immune; Boots ignore status on entry.
    """
    if mon is None or battle is None:
        return 0.0
    if _has_boots(mon):
        return 0.0

    conds = _side_conds_for(mon, battle)
    ts = int(conds.get("toxicspikes", 0) or 0)
    if ts <= 0:
        return 0.0

    if not _is_grounded(mon):
        return 0.0

    # Poison absorbs (good) -> no burden (we handle bonus separately in switch_score)
    if _is_poison(mon):
        return 0.0
    # Steel immune
    if _is_steel(mon):
        return 0.0

    # 1 layer = poison, 2 layers = toxic; treat toxic as higher long-run burden
    return 0.14 if ts == 1 else 0.24

def _sticky_web_burden(mon: Any, battle: Any) -> float:
    """
    Soft penalty for Sticky Web (speed control). Boots ignore; Flying not affected.
    """
    if mon is None or battle is None:
        return 0.0
    if _has_boots(mon):
        return 0.0

    conds = _side_conds_for(mon, battle)
    web = int(conds.get("stickyweb", 0) or 0)
    if web <= 0:
        return 0.0
    if not _is_grounded(mon):
        return 0.0

    return 0.10

def _team_hazard_burden(battle: Any, *, our_side: bool) -> float:
    """
    Aggregate hazard burden for a side.
    Returns a roughly normalized number in ~[0, 2].
    """
    if battle is None:
        return 0.0

    team = battle.team.values() if our_side else battle.opponent_team.values()
    mons = [m for m in team if m is not None and not getattr(m, "fainted", False)]
    if not mons:
        return 0.0

    total = 0.0
    for m in mons:
        total += _hazard_entry_damage_frac(m, battle)
        total += _toxic_spikes_burden(m, battle)
        total += _sticky_web_burden(m, battle)

    # Mild normalization by alive count (still increases with larger teams, but not linearly)
    avg = total / float(len(mons))
    return float(min(2.0, total * 0.35 + avg * 0.65))

def hazard_removal_value(move: Any, battle: Any, ctx: EvalContext) -> float:
    """
    Value of using a hazard-removal / hazard-control move this turn.

    Scales with:
      - how much hazards hurt our remaining team (boots-aware)
      - (for Defog/Court Change) the downside of removing OUR hazards on opponent
      - immediate survivability (don't waste a turn if we're getting deleted)
    """
    move_id = str(getattr(move, "id", "") or "").lower()
    if move_id not in {"defog", "rapidspin", "mortalspin", "courtchange", "tidyup"}:
        return 0.0

    me, opp = ctx.me, ctx.opp
    if me is None or opp is None:
        return 0.0

    our_burden = _team_hazard_burden(battle, our_side=True)
    opp_burden = _team_hazard_burden(battle, our_side=False)

    # If no relevant hazards, no value (Court Change can still be situational, but keep simple)
    if our_burden < 0.05 and move_id != "courtchange":
        return 0.0

    # Base benefit: removing hazards from our side
    net = our_burden

    # Defog removes hazards on BOTH sides -> losing our hazard advantage
    if move_id == "defog":
        net = our_burden - 0.75 * opp_burden

    # Court Change swaps hazards/screens -> benefit is roughly (our_burden - opp_burden)
    if move_id == "courtchange":
        net = our_burden - opp_burden

    # Tidy Up removes hazards and boosts; give a small constant extra
    tidy_bonus = 0.0
    if move_id == "tidyup":
        tidy_bonus = 12.0

    # Success probability / blockers
    success = 1.0
    if move_id in {"rapidspin", "mortalspin"}:
        # Rapid Spin fails vs Ghost; Mortal Spin still hits but may be blocked by Ghost interaction too
        # Use a conservative reduction if opponent is Ghost-type
        try:
            opp_types = getattr(opp, "types", None) or []
            if any(_type_name(t) == "ghost" for t in opp_types):
                success = 0.25 if move_id == "rapidspin" else 0.55
        except Exception:
            pass

    # Context: if we're about to die, spending a turn removing hazards is often bad
    my_hp = hp_frac(me)
    pressure = estimate_opponent_pressure(battle, ctx)
    imminent = (pressure.damage_to_me_frac >= my_hp * 0.90)

    context_mul = 1.0
    if imminent:
        context_mul *= 0.65
    else:
        # More valuable when we have more switching left
        our_remaining = remaining_count(battle.team)
        if our_remaining >= 4:
            context_mul *= 1.12
        elif our_remaining <= 2:
            context_mul *= 0.92

    # Convert net burden into score points (keep in same ballpark as other utility moves)
    value = 120.0 * max(0.0, net) * success * context_mul + tidy_bonus

    # Cap to prevent crazy spikes
    return float(max(0.0, min(160.0, value)))


def score_move(move: Any, battle: Any, ctx: EvalContext) -> float:
    """
    Score a move based on its effectiveness.
    
    Status moves: Use specialized status scoring
    Damage moves: Base damage + KO bonus + secondaries + chip synergy - penalties
    """
    me = ctx.me
    opp = ctx.opp
    if opp is None:
        return -100.0

    if move.category == MoveCategory.STATUS:
        # Status value + hazard control value
        return score_status_move(move, battle, ctx) + hazard_removal_value(move, battle, ctx)

    dmg_frac = float(estimate_damage_fraction(move, me, opp, battle))
    opp_hp = hp_frac(opp)

    # Base score from damage
    score = dmg_frac * 100.0
    
    # KO bonus
    ko_prob = ko_probability_from_fraction(dmg_frac, opp_hp)
    score += ko_bonus(ko_prob, slower=is_slower(me, opp))

    # Accuracy penalty
    acc = float(getattr(move, "accuracy", 1.0) or 1.0)
    acc = max(0.0, min(1.0, acc))
    if acc < 1.0:
        # Miss cost scales with move value (opportunity cost)
        # If we miss, we lose the value we would have gained
        miss_cost = score * 0.65  # Missing costs 65% of the move's value
        score = score * acc - (1.0 - acc) * miss_cost

    # Secondary effects
    score += score_secondaries(move, battle, ctx, ko_prob, dmg_frac=dmg_frac)

    # Chip synergy for plain attacks (enables revenge lines / Scald->Jet lines)
    score += 0.45 * chip_synergy_value(
        battle=battle,
        ctx=ctx,
        damage_dealt_frac=dmg_frac,
        after_status=None,
    )

    # Penalty for weak chip on healthy targets
    if opp_hp > 0.8 and dmg_frac < 0.18:
        score -= 8.0
    
    # Apply recoil penalty
    score -= _recoil_penalty(move, battle, ctx)
    
    # Apply PP conservation penalty
    score -= pp_conservation_penalty(move, battle, ctx)
    
    # Apply self-destruct penalty (considers KO prob and opponent threat)
    score -= _self_destruct_penalty(move, battle, ctx, ko_prob)
    # Apply drain move bonus
    score += _drain_move_bonus(move, battle, ctx)

    # Hazard control (Rapid Spin / Mortal Spin etc.)
    score += hazard_removal_value(move, battle, ctx)

    # Pivot moves bonus (U-turn / Volt Switch etc.)
    score += pivot_move_bonus(move, battle, ctx)

    return score