from __future__ import annotations

from typing import Any, Optional

from poke_env.battle import MoveCategory
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.pokemon_type import PokemonType

from bot.model.ctx import EvalContext
from bot.scoring.helpers import hp_frac, remaining_count
from bot.scoring.pressure import estimate_opponent_pressure
from bot.scoring.race import evaluate_race_for_move, DamageRace


def _estimate_damage_from_opponent(opponent: Any, target: Any, battle: Any) -> float:
    """
    Estimate the damage opponent would deal to target.

    NOTE:
      - Primary path uses poke-env Gen9 damage calculator (battle identifiers).
      - Falls back to opponent-pressure estimate if identifiers aren't available.
      - Final fallback is conservative.
    """
    if opponent is None or target is None:
        return 0.25

    try:
        from poke_env.calc.damage_calc_gen9 import calculate_damage

        best_avg_damage = 0.0

        for move in getattr(opponent, "moves", {}).values():
            if move is None:
                continue

            # Skip status moves
            if getattr(move, "category", None) == MoveCategory.STATUS:
                continue

            # Get identifiers
            opp_id = _get_pokemon_identifier(opponent, battle)
            target_id = _get_pokemon_identifier(target, battle)
            if opp_id is None or target_id is None:
                continue

            try:
                # Calculate damage (min/max roll)
                min_dmg, max_dmg = calculate_damage(
                    attacker_identifier=opp_id,
                    defender_identifier=target_id,
                    move=move,
                    battle=battle,
                    is_critical=False,
                )

                # Convert to fraction
                target_max_hp = getattr(target, "max_hp", None) or getattr(target, "stats", {}).get("hp", 100)
                if not target_max_hp or target_max_hp <= 0:
                    continue

                avg_dmg = (min_dmg + max_dmg) / 2.0
                dmg_frac = avg_dmg / float(target_max_hp)

                best_avg_damage = max(best_avg_damage, float(dmg_frac))

            except Exception:
                continue

        if best_avg_damage > 0:
            return float(best_avg_damage)

        # Fallback: use pressure model with a temp ctx (damage into `target`)
        try:
            temp_ctx = EvalContext(me=target, opp=opponent, battle=battle, cache={})
            pressure = estimate_opponent_pressure(battle, temp_ctx)
            return float(getattr(pressure, "damage_to_me_frac", 0.35) or 0.35)
        except Exception:
            return 0.35

    except Exception:
        return 0.25

def _estimate_damage_from_us(attacker: Any, defender: Any, battle: Any) -> float:
    """
    Estimate the damage we (attacker) would deal to defender.
    
    Returns: Fraction of defender's max HP (0.0 - 1.0+)
    """
    if attacker is None or defender is None:
        return 0.0
    
    try:
        from poke_env.calc.damage_calc_gen9 import calculate_damage
        from poke_env.battle import MoveCategory
        
        best_avg_damage = 0.0
        
        for move in getattr(attacker, "moves", {}).values():
            if move is None:
                continue
            
            # Skip status moves
            if getattr(move, "category", None) == MoveCategory.STATUS:
                continue
            
            # Get identifiers
            attacker_id = _get_pokemon_identifier(attacker, battle)
            defender_id = _get_pokemon_identifier(defender, battle)
            if attacker_id is None or defender_id is None:
                continue
            
            try:
                # Calculate damage (min/max roll)
                min_dmg, max_dmg = calculate_damage(
                    attacker_identifier=attacker_id,
                    defender_identifier=defender_id,
                    move=move,
                    battle=battle,
                    is_critical=False,
                )
                
                # Convert to fraction
                defender_max_hp = getattr(defender, "max_hp", None) or getattr(defender, "stats", {}).get("hp", 100)
                if not defender_max_hp or defender_max_hp <= 0:
                    continue
                
                avg_dmg = (min_dmg + max_dmg) / 2.0
                dmg_frac = avg_dmg / float(defender_max_hp)
                
                best_avg_damage = max(best_avg_damage, float(dmg_frac))
            
            except Exception:
                continue
        
        return float(best_avg_damage)
    
    except Exception:
        # Fallback: rough estimate based on types/stats
        # This is very crude but better than 0.0
        return 0.25

def _get_pokemon_identifier(pokemon: Any, battle: Any) -> Optional[str]:
    """Get battle identifier for a Pokemon."""
    if pokemon is None or battle is None:
        return None

    # Check player's team
    try:
        for identifier, pkmn in battle.team.items():
            if pkmn is pokemon:
                return identifier
    except Exception:
        pass

    # Check opponent's team
    try:
        for identifier, pkmn in battle.opponent_team.items():
            if pkmn is pokemon:
                return identifier
    except Exception:
        pass

    return None


def _type_multiplier(attack_type: PokemonType, defender: Any) -> float:
    """Return type effectiveness multiplier for attack_type into defender."""
    try:
        types = []
        t1 = getattr(defender, "type_1", None)
        t2 = getattr(defender, "type_2", None)
        if t1 is not None:
            types.append(t1)
        if t2 is not None and t2 != t1:
            types.append(t2)
        mult = 1.0
        for dt in types:
            mult *= float(attack_type.damage_multiplier(dt))
        return float(mult)
    except Exception:
        return 1.0


def _has_heavy_duty_boots(pokemon: Any) -> bool:
    item = str(getattr(pokemon, "item", "") or "").lower()
    return item in {"heavydutyboots", "heavy-dutyboots", "heavy-duty boots", "heavy_duty_boots"}


def _hazard_damage_frac_on_entry(pokemon: Any, battle: Any) -> float:
    """Estimate fraction of max HP lost on switch-in from damaging hazards on OUR side.

    Includes:
      - Stealth Rock (type effectiveness vs Rock)
      - Spikes (1-3 layers)
    Excludes:
      - Toxic Spikes (status), Sticky Web (speed) for now
    Ignores abilities (e.g., Levitate) for now, except Flying-type immunity to Spikes.
    Heavy-Duty Boots => 0.
    """
    if pokemon is None or battle is None:
        return 0.0
    if _has_heavy_duty_boots(pokemon):
        return 0.0

    try:
        side = getattr(battle, "side_conditions", None) or {}
    except Exception:
        side = {}

    dmg = 0.0

    # Stealth Rock: 1/8 * rock effectiveness
    if SideCondition.STEALTH_ROCK in side:
        dmg += (1.0 / 8.0) * _type_multiplier(PokemonType.ROCK, pokemon)

    # Spikes: grounded only, layers: 1 => 1/8, 2 => 1/6, 3 => 1/4
    layers = int(side.get(SideCondition.SPIKES, 0) or 0)
    if layers > 0:
        try:
            t1 = getattr(pokemon, "type_1", None)
            t2 = getattr(pokemon, "type_2", None)
            is_flying = (t1 == PokemonType.FLYING) or (t2 == PokemonType.FLYING)
        except Exception:
            is_flying = False

        if not is_flying:
            layers = min(3, max(0, layers))
            spikes_table = {1: 1.0/8.0, 2: 1.0/6.0, 3: 1.0/4.0}
            dmg += spikes_table.get(layers, 0.0)

    return float(max(0.0, min(1.0, dmg)))  # cap at 100%


def _switch_total_damage_on_entry(switch_target: Any, opponent: Any, battle: Any) -> float:
    """Total expected immediate damage fraction upon switching in: hazards + opponent's best hit."""
    if switch_target is None or opponent is None:
        return 0.0
    hazard = _hazard_damage_frac_on_entry(switch_target, battle)
    hit = _estimate_damage_from_opponent(opponent, switch_target, battle)
    return float(hazard + hit)



def _toxic_spikes_layers_on_our_side(battle: Any) -> int:
    """Number of Toxic Spikes layers on OUR side (0-2)."""
    if battle is None:
        return 0
    try:
        side = getattr(battle, "side_conditions", None) or {}
    except Exception:
        side = {}
    # Accept both enum and string keys for robustness
    layers = 0
    try:
        layers = int(side.get(SideCondition.TOXIC_SPIKES, 0) or 0)
    except Exception:
        layers = 0
    if layers == 0:
        try:
            layers = int(side.get("toxicspikes", 0) or 0)
        except Exception:
            layers = 0
    return max(0, min(2, int(layers)))


def _is_grounded(mon: Any) -> bool:
    """Grounded proxy: not Flying-type. (Ignores Levitate for now.)"""
    if mon is None:
        return False
    try:
        t1 = getattr(mon, "type_1", None)
        t2 = getattr(mon, "type_2", None)
        return not ((t1 == PokemonType.FLYING) or (t2 == PokemonType.FLYING))
    except Exception:
        return True


def _absorbs_toxic_spikes(mon: Any) -> bool:
    """Poison-type grounded mons absorb Toxic Spikes on switch-in."""
    if mon is None:
        return False
    if not _is_grounded(mon):
        return False
    try:
        t1 = getattr(mon, "type_1", None)
        t2 = getattr(mon, "type_2", None)
        return (t1 == PokemonType.POISON) or (t2 == PokemonType.POISON)
    except Exception:
        return False


def _toxic_spikes_absorb_bonus(mon: Any, battle: Any) -> float:
    """Bonus for clearing Toxic Spikes by switching in a grounded Poison-type."""
    layers = _toxic_spikes_layers_on_our_side(battle)
    if layers <= 0:
        return 0.0
    if not _absorbs_toxic_spikes(mon):
        return 0.0
    # 1 layer: poison prevention; 2 layers: toxic prevention (bigger)
    return float(20.0 + 10.0 * (layers - 1))
def _matchup_score(mon: Any, opponent: Any, battle: Any) -> float:
    """
    Evaluate how good mon's matchup is against opponent.
    Positive = good matchup, Negative = bad matchup
    """
    if mon is None or opponent is None:
        return 0.0

    score = 0.0

    # Defensive evaluation - how much damage does our mon take?
    damage_taken = _estimate_damage_from_opponent(opponent, mon, battle)

    if damage_taken < 0.15:
        score += 40  # Walls opponent completely
    elif damage_taken < 0.25:
        score += 25  # Tanks well
    elif damage_taken < 0.40:
        score += 0   # Neutral
    elif damage_taken < 0.60:
        score -= 20  # Takes heavy damage
    elif damage_taken < 0.90:
        score -= 40  # Near OHKO
    else:
        score -= 60  # OHKO'd

    # Offensive evaluation - how much damage can our mon deal?
    damage_dealt = _estimate_damage_from_opponent(mon, opponent, battle)

    if damage_dealt > 0.80:
        score += 40  # Threatens OHKO
    elif damage_dealt > 0.50:
        score += 25  # Threatens 2HKO
    elif damage_dealt > 0.30:
        score += 10  # Decent damage
    elif damage_dealt < 0.15:
        score -= 20  # Can't threaten

    # Speed control (very light)
    try:
        mon_speed = (mon.base_stats or {}).get("spe", 80)
        opp_speed = (opponent.base_stats or {}).get("spe", 80)

        if mon_speed >= opp_speed * 1.1:
            score += 15  # Outspeeds
        elif mon_speed <= opp_speed * 0.9:
            score -= 10  # Slower
    except Exception:
        pass

    return float(score)


def _danger_urgency(current_mon: Any, opponent: Any, battle: Any) -> float:
    """
    How urgently do we need to switch out?
    High values = urgent switch needed
    """
    if current_mon is None or opponent is None:
        return 0.0

    damage = _estimate_damage_from_opponent(opponent, current_mon, battle)
    hp = hp_frac(current_mon)

    # Immediate danger levels
    if damage >= hp * 0.95:
        return 80  # OHKO - urgent!
    elif damage >= hp * 0.85:
        return 65  # Near OHKO
    elif damage >= hp * 0.50:
        return 40  # 2HKO
    elif damage >= hp * 0.33:
        return 20  # 3HKO
    elif damage >= hp * 0.25:
        return 10  # 4HKO
    else:
        return 0   # Tanking fine


def _switch_in_penalty(switch_target: Any, opponent: Any, battle: Any) -> float:
    """
    Penalty for taking damage on switch-in.
    
    CRITICAL FIX: Now checks if we SURVIVE the switch-in, not just raw damage.
    
    Returns higher penalty if:
    - We get KO'd on switch (HUGE penalty)
    - We're left in KO range (big penalty)
    - We take heavy damage but survive (moderate penalty)
    
    Args:
        switch_target: Pokemon we're switching in
        opponent: Current opponent active
        battle: Battle state
    
    Returns:
        Penalty value (higher = worse switch)
    """
    if switch_target is None or opponent is None:
        return 0.0
    
    # Calculate total damage on switch-in
    hit_damage = _estimate_damage_from_opponent(opponent, switch_target, battle)
    hazard_damage = _hazard_damage_frac_on_entry(switch_target, battle)
    total_damage = hit_damage + hazard_damage
    
    # Get current HP of switch target
    current_hp = hp_frac(switch_target)
    
    # Calculate HP after switch
    hp_after = current_hp - total_damage
    
    # print(f"  [SWITCH PENALTY] {opponent.species} → {switch_target.species}")
    # print(f"  Current HP: {current_hp:.1%}")
    # print(f"  Hit damage: {hit_damage:.1%}")
    # print(f"  Hazard damage: {hazard_damage:.1%}")
    # print(f"  Total damage: {total_damage:.1%}")
    # print(f"  HP after switch: {hp_after:.1%}")
    
    # Do we survive?
    if hp_after <= 0.0:
        print(f"  → KO'd on switch! Penalty: 100")
        return 100.0
    
    # Are we left in immediate KO range?
    if hp_after <= 0.15:
        print(f"  → In KO range! Penalty: 70")
        return 70.0  # Very bad - likely to die next turn
    
    # Are we left weak but alive?
    if hp_after <= 0.30:
        penalty = 50.0 + (total_damage * 20)  # 50-70 range
        print(f"  → Left weak. Penalty: {penalty:.1f}")
        return penalty
    
    # We survive comfortably - penalty scales with damage taken
    if total_damage >= 0.60:
        penalty = 35.0  # Heavy damage but we're healthy enough
    elif total_damage >= 0.45:
        penalty = 25.0  # Significant damage
    elif total_damage >= 0.30:
        penalty = 15.0  # Moderate damage
    elif total_damage >= 0.20:
        penalty = 8.0   # Light damage
    elif total_damage >= 0.10:
        penalty = 3.0   # Chip damage
    else:
        penalty = 0.0   # Negligible
    
    print(f"  → Survivable damage. Penalty: {penalty:.1f}")
    return penalty


def _speed_control_value(switch_target: Any, opponent: Any, battle: Any, ctx: Any = None) -> float:
    """
    Value of gaining speed control by switching.
    
    Being faster than the opponent lets you:
    - Attack first (can KO before they act)
    - Set up safely (they can't revenge kill)
    - Force switches (they have to respect our speed)
    
    Args:
        switch_target: Pokemon we're switching in
        opponent: Current opponent active
        battle: Battle state
        ctx: Optional EvalContext for boost information
    
    Returns:
        Bonus value (higher = better)
    """
    if switch_target is None or opponent is None:
        return 0.0
    
    try:
        # Get base speeds
        target_base_speed = getattr(switch_target, "base_stats", {}).get("spe", 80)
        opp_base_speed = getattr(opponent, "base_stats", {}).get("spe", 80)
        
        # Get speed boosts (if available from context)
        target_spe_boost = 0
        opp_spe_boost = 0
        
        if ctx is not None:
            try:
                # Try to get boosts from shadow state if available
                if hasattr(ctx, 'my_boosts'):
                    target_spe_boost = ctx.my_boosts.get(id(switch_target), {}).get('spe', 0)
                if hasattr(ctx, 'opp_boosts'):
                    opp_spe_boost = ctx.opp_boosts.get(id(opponent), {}).get('spe', 0)
            except Exception:
                pass
        
        # Calculate effective speeds (simplified boost multiplier)
        # +1 = 1.5x, +2 = 2x, +3 = 2.5x, etc.
        def speed_multiplier(boost: int) -> float:
            if boost >= 1:
                return 1.0 + (boost * 0.5)
            elif boost <= -1:
                return 1.0 / (1.0 + (abs(boost) * 0.5))
            return 1.0
        
        target_eff_speed = target_base_speed * speed_multiplier(target_spe_boost)
        opp_eff_speed = opp_base_speed * speed_multiplier(opp_spe_boost)
        
        # Check paralysis (halves speed)
        try:
            from poke_env.battle import Status
            if getattr(switch_target, 'status', None) == Status.PAR:
                target_eff_speed *= 0.5
            if getattr(opponent, 'status', None) == Status.PAR:
                opp_eff_speed *= 0.5
        except Exception:
            pass
        
        # Do we outspeed?
        if target_eff_speed <= opp_eff_speed:
            return 0.0  # We don't outspeed, no bonus
        
        # If we outspeed, value depends on how much damage we threaten
        our_damage = _estimate_damage_from_us(switch_target, opponent, battle)
        opp_hp = hp_frac(opponent)
        
        # Critical: Can we KO them before they act?
        if our_damage >= opp_hp * 0.95:
            return 40.0  # HUGE value - we get a free KO
        
        # Can we threaten significant damage?
        if our_damage >= 0.60:
            return 30.0  # Very strong - outspeed + big damage
        elif our_damage >= 0.45:
            return 20.0  # Good - outspeed + solid damage
        elif our_damage >= 0.30:
            return 12.0  # Decent - outspeed + moderate damage
        else:
            return 5.0   # Small bonus - just speed control
    
    except Exception:
        return 0.0


def _positioning_value(current_mon: Any, switch_target: Any, opponent: Any, battle: Any) -> float:
    """
    Value of gaining positional advantage by switching.
    
    High value when:
    - Current mon is forced out anyway (switch is "free")
    - Switch-in forces opponent to switch (we gain momentum)
    - We maintain or gain type advantage
    
    Args:
        current_mon: Current active Pokemon
        switch_target: Pokemon we're switching in
        opponent: Current opponent active
        battle: Battle state
    
    Returns:
        Bonus value (higher = better)
    """
    if current_mon is None or switch_target is None or opponent is None:
        return 0.0
    
    from bot.scoring.switch_score import _estimate_damage_from_opponent, _matchup_score
    
    score = 0.0
    
    # Is current mon forced out anyway? (Switch is "free")
    current_hp = hp_frac(current_mon)
    damage_if_stay = _estimate_damage_from_opponent(opponent, current_mon, battle)
    
    if damage_if_stay >= current_hp * 0.95:
        score += 25.0  # Switch is free - staying dies anyway
        # print(f"  [POSITIONING] Free switch - current mon dies if stays (+25)")
    elif damage_if_stay >= current_hp * 0.75:
        score += 15.0  # Likely forced out next turn
        # print(f"  [POSITIONING] Likely forced out (+15)")
    
    # Do we force opponent to switch?
    target_damage_to_opp = _estimate_damage_from_us(switch_target, opponent, battle)
    opp_hp = hp_frac(opponent)
    
    if target_damage_to_opp >= opp_hp * 0.90:
        score += 20.0  # We threaten KO, they likely switch
        # print(f"  [POSITIONING] Force opponent switch - threaten KO (+20)")
    elif target_damage_to_opp >= 0.60:
        score += 12.0  # We threaten heavy damage
        # print(f"  [POSITIONING] Force opponent switch - heavy damage (+12)")
    
    # Do we maintain/gain type advantage?
    try:
        target_matchup = _matchup_score(switch_target, opponent, battle)
        current_matchup = _matchup_score(current_mon, opponent, battle)
        
        if target_matchup > 50 and target_matchup > current_matchup:
            score += 10.0  # Strong advantage gained
            # print(f"  [POSITIONING] Gain type advantage (+10)")
        elif target_matchup > 30:
            score += 5.0  # Maintain advantage
            # print(f"  [POSITIONING] Maintain advantage (+5)")
    except Exception:
        pass
    
    return score


def _setup_danger(mon: Any, opponent: Any, battle: Any, ctx: EvalContext) -> float:
    """
    Penalty if opponent can setup on mon.
    Positive = opponent can setup (bad for us)
    """
    if mon is None or opponent is None:
        return 0.0

    pressure = estimate_opponent_pressure(battle, ctx)
    setup_prob = float(getattr(pressure, "setup_prob", 0.0) or 0.0)

    if setup_prob < 0.3:
        return 0.0  # Opponent probably doesn't have setup

    # Check if mon can threaten opponent (prevents setup)
    damage_dealt = _estimate_damage_from_opponent(mon, opponent, battle)

    if damage_dealt > 0.60:
        return -15  # We threaten opponent, hard to setup
    elif damage_dealt > 0.40:
        return 0    # Neutral
    elif damage_dealt > 0.20:
        return 15 * setup_prob  # Opponent might try to setup
    else:
        return 30 * setup_prob  # Opponent can setup freely


def _win_condition_value(switch_target: Any, battle: Any) -> float:
    """
    Bonus for preserving important Pokemon.
    Positive = preserve this mon (don't risk it)
    """
    if switch_target is None:
        return 0.0

    value = 0.0

    # High HP Pokemon are more valuable
    hp = hp_frac(switch_target)
    if hp > 0.8:
        value += 10

    # Don't risk your last Pokemon unnecessarily
    our_remaining = remaining_count(battle.team)
    if our_remaining <= 2:
        value += 20

    return float(value)


def _best_race_for_mon_vs_opp(battle: Any, ctx: EvalContext, mon: Any, opp: Any) -> DamageRace:
    """
    One-step lookahead: evaluate whether `mon` can win the damage race vs `opp`
    (using mon's best available damaging move).
    """
    if mon is None or opp is None:
        return DamageRace(99.0, 99.0, "CLOSE", 0.0, 0)

    best: Optional[DamageRace] = None
    rank = {"WINNING": 2, "CLOSE": 1, "LOSING": 0}

    for mv in (getattr(mon, "moves", None) or {}).values():
        if mv is None:
            continue
        if getattr(mv, "category", None) == MoveCategory.STATUS:
            continue

        try:
            r = evaluate_race_for_move(battle, ctx, mv, me_override=mon, opp_override=opp)
        except TypeError:
            r = evaluate_race_for_move(battle, ctx, mv)

        if best is None:
            best = r
            continue

        if rank.get(r.state, 1) > rank.get(best.state, 1):
            best = r
        elif rank.get(r.state, 1) == rank.get(best.state, 1):
            if (r.tko_opp - r.ttd_me) < (best.tko_opp - best.ttd_me):
                best = r

    return best if best is not None else DamageRace(99.0, 99.0, "CLOSE", 0.0, 0)

def score_switch(pokemon: Any, battle: Any, ctx: Any) -> float:
    if pokemon is None or getattr(pokemon, "fainted", False):
        return -999.0
    
    current_mon = ctx.me
    opponent = ctx.opp
    
    if current_mon is None or opponent is None:
        return 10.0 * hp_frac(pokemon)
    
    if pokemon is current_mon:
        return -999.0
    
    # Matchup differential
    current_matchup = _matchup_score(current_mon, opponent, battle)
    new_matchup = _matchup_score(pokemon, opponent, battle)
    matchup_diff = new_matchup - current_matchup
    
    # Danger urgency (how badly we need to switch)
    danger = _danger_urgency(current_mon, opponent, battle)
    
    switch_penalty = _switch_in_penalty(pokemon, opponent, battle)
    
    # Setup danger differential
    current_setup_risk = _setup_danger(current_mon, opponent, battle, ctx)
    new_setup_risk = _setup_danger(pokemon, opponent, battle, ctx)
    setup_diff = current_setup_risk - new_setup_risk
    
    # Win condition preservation
    preserve_value = _win_condition_value(pokemon, battle)
    
    speed_value = _speed_control_value(pokemon, opponent, battle, ctx)
    
    positioning_value = _positioning_value(current_mon, pokemon, opponent, battle)
    
    # Race improvement (existing code)
    try:
        from bot.scoring.race import _best_race_for_mon_vs_opp
        now_race = _best_race_for_mon_vs_opp(battle, ctx, current_mon, opponent)
        in_race = _best_race_for_mon_vs_opp(battle, ctx, pokemon, opponent)
        
        if now_race.state in ("LOSING", "CLOSE") and in_race.state == "WINNING":
            preserve_value += 2.5
        elif now_race.state == "LOSING" and in_race.state == "CLOSE":
            preserve_value += 1.0
        elif now_race.state == "WINNING" and in_race.state == "LOSING":
            preserve_value -= 1.5
    except Exception:
        pass
    
    score = (
        matchup_diff * 40          # Matchup improvement
        + danger * 10              # Urgency to switch
        - switch_penalty * 50      # Damage taken (FIXED)
        + setup_diff * 25          # Setup danger reduction
        + preserve_value * 10      # Win-con preservation
        + speed_value              # Speed control (NEW)
        + positioning_value        # Positioning (NEW)
    )
    
    # Toxic Spikes absorption
    score += _toxic_spikes_absorb_bonus(pokemon, battle)
    
    # Regenerator ability
    if str(getattr(pokemon, "ability", "")).lower() == "regenerator":
        score += 15
    
    # Intimidate ability (vs physical attackers)
    if str(getattr(pokemon, "ability", "")).lower() == "intimidate":
        pressure = estimate_opponent_pressure(battle, ctx)
        physical_prob = float(getattr(pressure, "physical_prob", 0.0) or 0.0)
        if physical_prob > 0.5:
            score += 20 * physical_prob

    # Low danger penalty (don't switch unnecessarily)
    if danger < 40:
        score -= 15
    
    # Late game logic
    our_remaining = remaining_count(battle.team)
    opp_remaining = remaining_count(battle.opponent_team)
    
    if our_remaining <= 2:
        # Late game - be much more conservative
        from bot.scoring.switch_score import _switch_total_damage_on_entry
        total_damage = _switch_total_damage_on_entry(pokemon, opponent, battle)
        
        if total_damage > 0.50:
            score -= 50  # Heavy penalty for risky late switches
        elif total_damage > 0.35:
            score -= 30
        
        # Only switch if matchup improvement is significant
        if matchup_diff < 30:
            score -= 35  # Don't make marginal switches late
    
    elif our_remaining <= 2 and opp_remaining <= 2:
        # Both in late game
        if matchup_diff < 20:
            score -= 20
    
    # print(f"\n=== IMPROVED SWITCH SCORE: {pokemon.species} ===")
    # print(f"Matchup diff: {matchup_diff:.1f} × 40 = {matchup_diff * 40:.1f}")
    # print(f"Danger: {danger:.1f} × 10 = {danger * 10:.1f}")
    # print(f"Penalty: {switch_penalty:.1f} × 50 = {-switch_penalty * 50:.1f}")
    # print(f"Speed value: {speed_value:.1f}")
    # print(f"Positioning: {positioning_value:.1f}")
    # print(f"Total: {score:.1f}")
    
    return float(score)

def _slow_pivot_value(current_mon: Any, switch_target: Any, opponent: Any, battle: Any) -> float:
    """
    Value of being able to slow pivot vs hard switch.
    """
    if current_mon is None or switch_target is None or opponent is None:
        return 0.0

    try:
        my_speed = (current_mon.base_stats or {}).get("spe", 80)
        opp_speed = (opponent.base_stats or {}).get("spe", 80)
        target_speed = (switch_target.base_stats or {}).get("spe", 80)
    except Exception:
        return 0.0

    if my_speed >= opp_speed:
        return 0.0

    hazard_damage = _hazard_damage_frac_on_entry(switch_target, battle)
    opp_hit_on_switch = _estimate_damage_from_opponent(opponent, switch_target, battle)
    total_hard_switch = hazard_damage + opp_hit_on_switch
    if total_hard_switch < 0.3:
        return 0.0

    my_hp = hp_frac(current_mon)
    damage_to_me = _estimate_damage_from_opponent(opponent, current_mon, battle)
    if damage_to_me >= my_hp * 0.95:
        return 0.0

    target_matchup = _matchup_score(switch_target, opponent, battle)
    if target_matchup < 20:
        return 0.0

    value = opp_hit_on_switch * 80

    target_hp = hp_frac(switch_target)
    if target_hp < 0.6:
        value *= 1.3

    if target_speed > opp_speed:
        value += 15

    if opp_hit_on_switch > 0.7:
        value *= 1.4
    elif opp_hit_on_switch > 0.5:
        value *= 1.2

    return float(value)


def _pivot_order(me: Any, opp: Any, move: Any) -> int:
    """Determine effective move order for a pivot move.

    Returns:
      +1: we pivot before opponent acts (fast pivot)
      -1: opponent acts before we pivot (slow pivot)
       0: unclear / near tie

    Uses base speed only with a small deadzone. Priority overrides speed.
    """
    if me is None or opp is None:
        return 0

    prio = int(getattr(move, "priority", 0) or 0)
    if prio > 0:
        return +1
    if prio < 0:
        return -1

    try:
        ms = float((getattr(me, "base_stats", None) or {}).get("spe", 80))
        os = float((getattr(opp, "base_stats", None) or {}).get("spe", 80))
    except Exception:
        return 0

    if ms >= os * 1.05:
        return +1
    if os >= ms * 1.05:
        return -1
    return 0


def _fast_pivot_value(current_mon: Any, switch_target: Any, opponent: Any, battle: Any) -> float:
    """Extra value of *fast* pivoting (we act first).

    Fast pivot is valuable mainly when:
      - staying in is dangerous (opponent threatens big damage / KO)
      - and pivoting shifts that hit onto a much safer switch_target

    Unlike slow pivot, it does NOT avoid the hit on the switch_target.
    """
    if current_mon is None or switch_target is None or opponent is None:
        return 0.0

    target_matchup = _matchup_score(switch_target, opponent, battle)
    if target_matchup < 10:
        return 0.0

    my_hp = hp_frac(current_mon)
    dmg_to_me = _estimate_damage_from_opponent(opponent, current_mon, battle)
    dmg_to_target = _estimate_damage_from_opponent(opponent, switch_target, battle)
    hazard_damage = _hazard_damage_frac_on_entry(switch_target, battle)
    target_cost = dmg_to_target + hazard_damage

    if dmg_to_me < 0.35 and dmg_to_me < my_hp * 0.75:
        return 0.0

    avoided = max(0.0, dmg_to_me - target_cost)
    if avoided < 0.12:
        return 0.0

    value = avoided * 90.0

    if dmg_to_me >= my_hp * 0.95 and target_cost < 0.70:
        value += 25.0

    danger = _danger_urgency(current_mon, opponent, battle)
    if danger > 60:
        value *= 1.25
    elif danger > 40:
        value *= 1.10

    return float(value)


def _switch_quality_for_pivot(current_mon: Any, target: Any, opponent: Any, battle: Any, ctx: EvalContext) -> float:
    """
    Small, bounded "is this a good mon to bring in AFTER a pivot" score.

    IMPORTANT:
      - Do NOT reuse score_switch() here: it includes urgency/preserve/endgame scaling
        intended for HARD SWITCH decisions.
      - This is only to measure follow-up quality.
    """
    if current_mon is None or target is None or opponent is None:
        return 0.0

    cur_match = _matchup_score(current_mon, opponent, battle)
    new_match = _matchup_score(target, opponent, battle)
    matchup_diff = new_match - cur_match

    switch_penalty = _switch_in_penalty(target, opponent, battle)

    try:
        cur_setup = _setup_danger(current_mon, opponent, battle, ctx)
        new_setup = _setup_danger(target, opponent, battle, ctx)
        setup_diff = cur_setup - new_setup
    except Exception:
        setup_diff = 0.0

    q = (matchup_diff * 1.0) - (switch_penalty * 0.8) + (setup_diff * 0.5)

    # Clearing Toxic Spikes is strong utility; keep it bounded here.
    layers = _toxic_spikes_layers_on_our_side(battle)
    if layers > 0 and _absorbs_toxic_spikes(target):
        q += 15.0 + 10.0 * (layers - 1)

    return max(-60.0, min(60.0, float(q)))


PIVOT_MOVES = {
    "uturn",
    "voltswitch",
    "flipturn",
    "partingshot",
    "batonpass",
    "chillyreception",
    "shedtail",
    "teleport",
}


def is_pivot_move(move: Any) -> bool:
    """Check if a move is a pivot move."""
    move_id = str(getattr(move, "id", "")).lower()
    return move_id in PIVOT_MOVES


def pivot_move_bonus(move: Any, battle: Any, ctx: EvalContext) -> float:
    """
    Calculate bonus for pivot moves.

    Pivot move bonus should be:
      - modest, stable
      - NOT derived from full hard-switch score
      - include either fast OR slow pivot safety (mutually exclusive)
    """
    if not is_pivot_move(move):
        return 0.0

    me = ctx.me
    opp = ctx.opp
    if me is None or opp is None:
        return 0.0

    bonus = 0.0

    # Momentum
    bonus += 15.0

    # Scout
    bonus += 10.0

    # Best follow-up switch-in quality (bounded; NOT score_switch)
    best_q = -999.0
    best_switch_target = None

    for teammate in battle.team.values():
        if teammate is None or teammate.fainted or teammate is me:
            continue

        q = _switch_quality_for_pivot(me, teammate, opp, battle, ctx)
        if q > best_q:
            best_q = q
            best_switch_target = teammate

    best_q = max(-60.0, min(60.0, best_q))
    if best_q >= 0:
        bonus += best_q * 0.5
    else:
        bonus += best_q * 1.0

    # Fast vs slow pivot value (mutually exclusive)
    order = _pivot_order(me, opp, move)
    if best_switch_target is not None:
        if order == +1:
            bonus += _fast_pivot_value(me, best_switch_target, opp, battle)
        elif order == -1:
            bonus += _slow_pivot_value(me, best_switch_target, opp, battle)

    # Risk penalty applies only when opponent acts before pivot (slow pivot)
    danger = _danger_urgency(me, opp, battle)
    if order == -1:
        if danger > 60:
            bonus -= 30
        elif danger > 40:
            bonus -= 15

    # Choice item synergy
    my_item = str(getattr(me, "item", "")).lower()
    if my_item in ["choiceband", "choicescarf", "choicespecs"]:
        bonus += 20.0

    return float(bonus)
