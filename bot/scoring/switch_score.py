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
    High values = bad switch (takes heavy damage)
    """
    if switch_target is None or opponent is None:
        return 0.0

    damage = _estimate_damage_from_opponent(opponent, switch_target, battle)
    print(f"  [SWITCH PENALTY] {opponent.species} → {switch_target.species}")
    print(f"  Predicted damage: {damage:.1%}")

    # Penalty scales with damage
    if damage >= 0.80:
        return 80  # Switch-in nearly dies
    elif damage >= 0.50:
        return 50  # Heavy damage
    elif damage >= 0.30:
        return 20  # Moderate damage
    elif damage >= 0.15:
        return 5   # Light chip
    else:
        return 0   # Minimal damage


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


def score_switch(pokemon: Any, battle: Any, ctx: EvalContext) -> float:
    """
    Score switching to a specific Pokemon.
    """
    if pokemon is None or getattr(pokemon, "fainted", False):
        return -999.0

    current_mon = ctx.me
    opponent = ctx.opp

    if current_mon is None or opponent is None:
        return 10.0 * hp_frac(pokemon)

    if pokemon is current_mon:
        return -999.0

    current_matchup = _matchup_score(current_mon, opponent, battle)
    new_matchup = _matchup_score(pokemon, opponent, battle)
    matchup_diff = new_matchup - current_matchup

    danger = _danger_urgency(current_mon, opponent, battle)
    switch_penalty = _switch_in_penalty(pokemon, opponent, battle)

    current_setup_risk = _setup_danger(current_mon, opponent, battle, ctx)
    new_setup_risk = _setup_danger(pokemon, opponent, battle, ctx)
    setup_diff = current_setup_risk - new_setup_risk

    preserve_value = _win_condition_value(pokemon, battle)

    # Race improvement shaping (small)
    try:
        now_race = _best_race_for_mon_vs_opp(battle, ctx, current_mon, opponent)
        in_race = _best_race_for_mon_vs_opp(battle, ctx, pokemon, opponent)

        if now_race.state in ("LOSING", "CLOSE") and in_race.state == "WINNING":
            preserve_value += 2.5  # => +25 after *10
        elif now_race.state == "LOSING" and in_race.state == "CLOSE":
            preserve_value += 1.0  # => +10
        elif now_race.state == "WINNING" and in_race.state == "LOSING":
            preserve_value -= 1.5  # => -15
    except Exception:
        pass

    score = (
        matchup_diff * 40
        + danger * 10
        - switch_penalty * 50
        + setup_diff * 25
        + preserve_value * 10
    )


    # Toxic Spikes absorption bonus (grounded Poison-types clear them on entry)
    score += _toxic_spikes_absorb_bonus(pokemon, battle)
    # Special cases
    if str(getattr(pokemon, "ability", "")).lower() == "regenerator":
        score += 15

    if str(getattr(pokemon, "ability", "")).lower() == "intimidate":
        pressure = estimate_opponent_pressure(battle, ctx)
        physical_prob = float(getattr(pressure, "physical_prob", 0.0) or 0.0)
        if physical_prob > 0.5:
            score += 20 * physical_prob

    if danger < 40:
        score -= 15

    our_remaining = remaining_count(battle.team)
    opp_remaining = remaining_count(battle.opponent_team)
    if our_remaining <= 2 and opp_remaining <= 2:
        if matchup_diff < 20:
            score -= 20

    # Optional debug (leave on for now; pivot no longer calls score_switch)
    print(f"\n=== SWITCH DEBUG: {pokemon.species} ===")
    print(f"Matchup: {matchup_diff:.1f} × 40 = {matchup_diff * 40:.1f}")
    print(f"Danger: {danger:.1f} × 10 = {danger * 10:.1f}")
    print(f"Penalty: {switch_penalty:.1f} × 50 = {-switch_penalty * 50:.1f}")
    print(f"Total: {score:.1f}")

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
