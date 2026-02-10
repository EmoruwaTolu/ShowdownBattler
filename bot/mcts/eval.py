import math
from typing import Any

from poke_env.battle import Status
from bot.model.opponent_model import build_opponent_belief
from bot.scoring.race import evaluate_race_for_move


def _tanh01(x: float) -> float:
    # maps to (-1, 1)
    return math.tanh(float(x))


def _team_hp_sum(hp_map: dict[int, float]) -> float:
    return float(sum(max(0.0, min(1.0, v)) for v in hp_map.values()))

HAZARDS = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
REMOVAL = {"rapidspin", "defog"}
WEATHER_ABIL = {"drought", "drizzle", "sandstream", "snowwarning"}

def candidate_role_weight(c) -> float:
    w = 1.0

    # wincon / high leverage roles
    if c.has_setup:
        w *= 1.10
    if c.has_priority:
        w *= 1.05

    # support / team enablers
    if any(m in HAZARDS for m in c.moves):
        w *= 1.06
    if any(m in REMOVAL for m in c.moves):
        w *= 1.04
    if any(a in WEATHER_ABIL for a in c.abilities):
        w *= 1.12

    return w

def expected_role_weight_for_mon(mon, gen: int) -> float:
    """
    Uses your RandBats DB to infer the mon's role weight (belief-averaged).
    Works for BOTH our mons and opponent mons (we know all our info; opp is filtered by revealed info).
    """
    belief = build_opponent_belief(mon, gen)  # works fine for "any mon snapshot"
    return sum(candidate_role_weight(c) * p for (c, p) in belief.as_distribution())

def low_hp_multiplier(hp: float, role_w: float) -> float:
    # important mons (role_w > 1) get punished more for being low
    if hp < 0.20:
        return 0.35 if role_w > 1.06 else 0.45
    if hp < 0.35:
        return 0.55 if role_w > 1.06 else 0.65
    if hp < 0.55:
        return 0.80
    return 1.00

def boost_multiplier(boosts: dict, hp: float) -> float:
    if not boosts:
        return 1.00
    max_pos = max((v for v in boosts.values() if v > 0), default=0)
    if hp < 0.35:
        max_pos = min(max_pos, 2)  # don't overvalue +6 at 20% HP
    if max_pos >= 4:
        return 1.18
    if max_pos >= 2:
        return 1.12
    if max_pos >= 1:
        return 1.06
    return 1.00

def team_value(team, hp_map, boosts_map, gen: int) -> float:
    total = 0.0
    for mon in team:
        hp = float(hp_map.get(id(mon), 0.0))
        if hp <= 0.0:
            continue

        role_w = expected_role_weight_for_mon(mon, gen)
        boosts = (boosts_map.get(id(mon), {}) if boosts_map else {}) or {}

        v = hp
        v *= role_w
        v *= boost_multiplier(boosts, hp)
        v *= low_hp_multiplier(hp, role_w)

        total += v
    return total

def opp_unseen_value(opp_known_count: int, opp_total: int = 6) -> float:
    #Assuming every mon we haven't seen has a value of one (should probably find out the avg "value" of each possible mon and replace)
    unseen = max(0, opp_total - opp_known_count)
    return unseen * 1.0 

def healthy_count(team, hp_map, thresh=0.55) -> int:
        return sum(1 for mon in team if float(hp_map.get(id(mon), 0.0)) >= thresh)

def evaluate_boosts(state: Any) -> float:
    """
    Evaluate stat boost advantage with DIMINISHING RETURNS.
    
    Going from +0 → +1 is worth more than +2 → +3, hence the diminishing returns function
    """
    my_active_id = id(state.my_active)
    opp_active_id = id(state.opp_active)
    
    my_boosts = state.my_boosts.get(my_active_id, {})
    opp_boosts = state.opp_boosts.get(opp_active_id, {})
    
    # Calculate boost value
    my_value = _calculate_boost_state_value(my_boosts)
    opp_value = _calculate_boost_state_value(opp_boosts)
    
    diff = my_value - opp_value
    return _tanh01(diff / 10.0)


def _calculate_boost_state_value(boosts: dict) -> float:
    """
    Calculate the value of a boost state with diminishing returns.
    
    Returns a value where each additional boost stage is worth less.
    """
    value = 0.0
    
    for stat in ['atk', 'spa', 'spe', 'def', 'spd']:
        stages = boosts.get(stat, 0)
        
        # Weight by stat importance
        if stat in ['atk', 'spa']:
            base_weight = 1.5
        elif stat == 'spe':
            base_weight = 1.2
        else:
            base_weight = 0.7
        
        # Apply diminishing returns
        if stages > 0:
            # Positive boosts with diminishing returns
            stage_value = 0.0
            for i in range(1, stages + 1):
                if i == 1:
                    stage_value += 1.0  # 1st stage: full value
                elif i == 2:
                    stage_value += 0.8  # 2nd stage: 80%
                elif i == 3:
                    stage_value += 0.6  # 3rd stage: 60%
                elif i == 4:
                    stage_value += 0.4  # 4th stage: 40%
                elif i == 5:
                    stage_value += 0.2  # 5th stage: 20%
                else:
                    stage_value += 0.1  # 6th stage: 10%
            
            value += stage_value * base_weight
        
        elif stages < 0:
            # Negative boosts (penalty)
            value += stages * base_weight
    
    return value

def evaluate_state(state: Any) -> float:
    """
    Returns a scalar value for MCTS backup: higher is better for us.
    Range ~[-1, +1].

    Uses:
      - Team HP advantage (anchor)
      - Best-move damage-race advantage (tempo)
      - Best switch option (escape hatch)
      - Small status shaping (BRN/PAR)

    Ideas being considered here:
      - Have our setup mons taken too much damage?
      - Have we weakened a big threat?
      - Do we have Pokémon that help others (Drought, hazards, removal)?
    """
    # Terminal: if a side has no HP left, hard value
    my_sum_raw = _team_hp_sum(state.my_hp)
    opp_sum_raw = _team_hp_sum(state.opp_hp)

    # If we're out of HP, it's terminal loss regardless of opponent info
    if my_sum_raw <= 1e-9:
        return -1.0

    # Only allow "terminal win" if opponent team is fully represented (or battle says finished)
    opp_known = len(getattr(state, "opp_team", []) or [])
    opp_total = int(getattr(state, "opp_team_size", 6) or 6)

    battle_finished = bool(getattr(state.battle, "finished", False))
    if opp_sum_raw <= 1e-9 and (battle_finished or opp_known >= opp_total):
        return +1.0

    gen = int(getattr(state.battle, "gen", 9) or 9)

    with state._patched_status(), state._patched_boosts():
        # --- Belief/role weighted team value ---
        my_value = team_value(state.my_team, state.my_hp, state.my_boosts, gen)
        opp_value_known = team_value(state.opp_team, state.opp_hp, state.opp_boosts, gen)

        # Unseen opponent slots: treat as alive resources (prevents 1v1 behavior)
        opp_unseen = max(0, opp_total - opp_known)
        # Start simple: each unseen slot ~= 1.0 value (full HP, no boosts)
        opp_value = opp_value_known + float(opp_unseen) * 1.0

        team_term = _tanh01((my_value - opp_value) / 1.2)

        # Numbers advantage (light)
        my_healthy = healthy_count(state.my_team, state.my_hp, 0.55)
        # assume unseen mons are healthy
        opp_healthy = healthy_count(state.opp_team, state.opp_hp, 0.55) + opp_unseen
        numbers_term = _tanh01((my_healthy - opp_healthy) / 1.5)

        # Best-move race advantage
        best_mv = None
        best_mv_score = -1e18
        for (kind, obj) in state.legal_actions():
            if kind != "move" or obj is None:
                continue
            if getattr(obj, "base_power", 0) <= 0:
                continue
            s = float(state.score_move_fn(obj, state.battle, state.ctx_me))
            if s > best_mv_score:
                best_mv_score = s
                best_mv = obj

        race_term = 0.0
        if best_mv is not None and state.ctx_me is not None:
            race = evaluate_race_for_move(state.battle, state.ctx_me, best_mv)
            race_term = _tanh01((race.ttd_me - race.tko_opp) / 1.5)

        # Escape hatch
        switch_term = 0.0
        if race_term < 0.0:
            best_sw_score = -1e18
            for p in state.my_team:
                if p is state.my_active:
                    continue
                if state.my_hp.get(id(p), 0.0) <= 0.0:
                    continue
                sc = float(state.score_switch_fn(p, state.battle, state.ctx_me))
                if sc > best_sw_score:
                    best_sw_score = sc
            switch_term = _tanh01(best_sw_score / 120.0)

        boost_term = evaluate_boosts(state)

        # Status shaping 
        status_term = 0.0

        # Will consider boosting these more if the status is valuable on that pokemon
        if state.opp_status.get(id(state.opp_active)) == Status.BRN:
            status_term += 0.10
        if state.opp_status.get(id(state.opp_active)) == Status.PAR:
            status_term += 0.06

        if state.my_status.get(id(state.my_active)) == Status.BRN:
            status_term -= 0.10
        if state.my_status.get(id(state.my_active)) == Status.PAR:
            status_term -= 0.06

        # Poison matters for "preserve my breaker" style
        if state.my_status.get(id(state.my_active)) in (Status.PSN, Status.TOX):
            status_term -= 0.05

        # --- Active preservation: discourage leaving important active too low ---
        my_active_hp = float(state.my_hp.get(id(state.my_active), 0.0))
        my_active_role = expected_role_weight_for_mon(state.my_active, gen)
        # only penalize hard if this looks like an important mon
        if my_active_role > 1.06:
            active_preserve = _tanh01((my_active_hp - 0.60) / 0.20)
        else:
            active_preserve = _tanh01((my_active_hp - 0.45) / 0.25)

    tempo_penalty = 0.04 * state.ply

    # Weights: anchor on team_term, keep tempo, keep escape hatch, keep boosts/status.
    value = (
        0.38 * team_term +
        0.10 * numbers_term +
        0.25 * race_term +
        0.10 * switch_term +
        0.10 * boost_term +
        0.07 * active_preserve +
        status_term
    ) - tempo_penalty

    return max(-1.0, min(1.0, float(value)))
