import math
import json
import os
import re
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

# Normalization that matches Showdown-ish move ids (e.g., "Stealth Rock" -> "stealthrock")
def _norm_id(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())

SETUP_MOVES = {_norm_id(x) for x in [
    "Swords Dance", "Nasty Plot", "Dragon Dance", "Calm Mind", "Bulk Up", "Quiver Dance",
    "Shell Smash", "Belly Drum", "Shift Gear", "Agility", "Tail Glow", "Coil", "Curse", "Growth",
]}
PRIORITY_MOVES = {_norm_id(x) for x in [
    "Extreme Speed", "Aqua Jet", "Mach Punch", "Ice Shard", "Sucker Punch", "Bullet Punch",
    "Shadow Sneak", "Quick Attack", "Vacuum Wave", "First Impression",
]}

def load_randbats_db() -> dict:
    """Best-effort loader for the gen9 randombattle set DB."""
    candidates = [
        os.getenv("RANDBATS_DB_PATH", ""),
        "gen9randombattle.json",
    ]
    for path in candidates:
        if not path:
            continue
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    return json.load(f)
        except Exception:
            pass
    return {}

RANDBATS_DB = load_randbats_db()

def avg_unseen_role_weight(db: dict) -> float:
    """Average candidate_role_weight over all (species, role) entries in the DB."""
    if not db:
        return 1.0

    total = 0.0
    count = 0

    for _, info in db.items():
        roles = (info or {}).get("roles", {}) or {}
        for role_name, role in roles.items():
            moves = {_norm_id(m) for m in (role or {}).get("moves", [])}
            abils = {_norm_id(a) for a in (role or {}).get("abilities", [])}

            has_setup = (any(m in SETUP_MOVES for m in moves) or ("setup" in (role_name or "").lower()))
            has_priority = any(m in PRIORITY_MOVES for m in moves)

            # Mirror candidate_role_weight() but using role info directly
            w = 1.0
            if has_setup:
                w *= 1.10
            if has_priority:
                w *= 1.05
            if any(m in HAZARDS for m in moves):
                w *= 1.06
            if any(m in REMOVAL for m in moves):
                w *= 1.04
            if any(a in WEATHER_ABIL for a in abils):
                w *= 1.12

            total += float(w)
            count += 1

    if count <= 0:
        return 1.0
    return float(total / count)

AVG_UNSEEN_ROLE_WEIGHT = avg_unseen_role_weight(RANDBATS_DB)

# Discount so unseen priors don't over-dominate real, known board state
UNSEEN_SLOT_DISCOUNT = 0.90

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
    """Belief-averaged role weight for a mon snapshot."""
    try:
        belief = build_opponent_belief(mon, gen)
        w = sum(candidate_role_weight(c) * p for (c, p) in belief.as_distribution())
        # keep it sane
        return float(max(0.80, min(1.35, w)))
    except Exception:
        return 1.0


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

def status_multiplier(status) -> float:
    if status in (Status.TOX, Status.PSN):
        return 0.85
    if status == Status.BRN:
        return 0.88
    if status == Status.PAR:
        return 0.93
    return 1.00

def team_value(team, hp_map, boosts_map, gen: int, status_map=None) -> float:
    total = 0.0
    status_map = status_map or {}

    for mon in team:
        hp = float(hp_map.get(id(mon), 0.0))
        if hp <= 0.0:
            continue

        role_w = expected_role_weight_for_mon(mon, gen)
        boosts = (boosts_map.get(id(mon), {}) if boosts_map else {}) or {}
        st = status_map.get(id(mon), None)

        v = hp
        v *= role_w
        v *= boost_multiplier(boosts, hp)
        v *= low_hp_multiplier(hp, role_w)
        v *= status_multiplier(st)

        total += v
    return total

def opp_unseen_value(opp_known_count: int, opp_total: int = 6) -> float:
    unseen = max(0, opp_total - opp_known_count)

    # In the early game lack of knowledge is more threatening, but late game we should trust the board more.
    if opp_known_count <= 1:
        d = 0.95
    elif opp_known_count <= 3:
        d = 0.90
    elif opp_known_count <= 5:
        d = 0.80
    else:
        d = 0.0

    return float(unseen * AVG_UNSEEN_ROLE_WEIGHT * d)

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

    6v6-aware shaping:
      - Belief-weighted team value (HP * role importance * boosts, with low-HP penalty)
      - Unseen opponent slot value (prevents "it's just a 1v1" behavior)
      - Numbers advantage
      - Local race/tempo (best damaging move)
      - Escape hatch (best switch) only when race is bad
      - Boost advantage (active vs active) with diminishing returns
      - Status shaping (small)
      - Active preservation term (small)
      - Progress term (prefer converting the lead)
      - Sac penalty when ahead (don’t throw away mons for no reason)
    """

    # --- Hard terminal-ish checks that should dominate everything ---
    my_sum_raw = _team_hp_sum(state.my_hp)
    opp_sum_raw = _team_hp_sum(state.opp_hp)

    if my_sum_raw <= 1e-9:
        return -1.0

    # Only allow "full win" if battle says finished OR we have full opponent roster represented.
    opp_known = len(getattr(state, "opp_team", []) or [])
    opp_total = int(getattr(state, "opp_team_size", 6) or 6)
    battle_finished = bool(getattr(state.battle, "finished", False))

    if opp_sum_raw <= 1e-9 and (battle_finished or opp_known >= opp_total):
        return +1.0

    # IMPORTANT SAFETY:
    # If our active is fainted but opponent active is not, this state is *immediately bad*
    # (and in your current simulator it can happen mid-turn due to move ordering).
    my_active_hp = float(state.my_hp.get(id(state.my_active), 0.0))
    opp_active_hp = float(state.opp_hp.get(id(state.opp_active), 0.0))
    if my_active_hp <= 0.0 and opp_active_hp > 0.0:
        # Don’t let boosts/race accidentally make this look good.
        # Still give *tiny* credit if we’re massively ahead in resources, but keep it very negative.
        # (You can tune 0.15 later.)
        lead_hint = _tanh01((my_sum_raw - opp_sum_raw) / 1.5)
        return max(-1.0, min(1.0, float(-0.90 + 0.15 * lead_hint)))

    gen = int(getattr(state.battle, "gen", 9) or 9)

    with state._patched_status(), state._patched_boosts():
        # --- Team value (belief-weighted) ---
        my_value = team_value(state.my_team, state.my_hp, state.my_boosts, gen)
        opp_value_known = team_value(state.opp_team, state.opp_hp, state.opp_boosts, gen)

        # Unseen opponent slots treated as alive resources.
        opp_unseen = max(0, opp_total - opp_known)
        opp_value = opp_value_known + opp_unseen_value(opp_known, opp_total)

        team_term = _tanh01((my_value - opp_value) / 1.2)

        # --- Numbers advantage (healthy count) ---
        my_healthy = healthy_count(state.my_team, state.my_hp, 0.55)
        opp_healthy = healthy_count(state.opp_team, state.opp_hp, 0.55) + opp_unseen
        numbers_term = _tanh01((my_healthy - opp_healthy) / 1.5)

        # --- Best-move race advantage (local tempo) ---
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

        # --- Escape hatch only when losing the race ---
        switch_term = 0.0
        if race_term < 0.0:
            best_sw_score = -1e18
            for p in state.my_team:
                if p is state.my_active:
                    continue
                if float(state.my_hp.get(id(p), 0.0)) <= 0.0:
                    continue
                sc = float(state.score_switch_fn(p, state.battle, state.ctx_me))
                if sc > best_sw_score:
                    best_sw_score = sc
            switch_term = _tanh01(best_sw_score / 120.0)

        # --- Boost term (active vs active) ---
        # If our active is extremely low, don’t let boosts dominate (they might be unusable).
        boost_term = evaluate_boosts(state)
        if my_active_hp < 0.20:
            boost_term *= 0.40
        elif my_active_hp < 0.35:
            boost_term *= 0.70

        # --- Status shaping (small, but “preserve breaker” aware) ---
        status_term = 0.0
        if state.opp_status.get(id(state.opp_active)) == Status.BRN:
            status_term += 0.10
        if state.opp_status.get(id(state.opp_active)) == Status.PAR:
            status_term += 0.06
        if state.my_status.get(id(state.my_active)) == Status.BRN:
            status_term -= 0.10
        if state.my_status.get(id(state.my_active)) == Status.PAR:
            status_term -= 0.06
        if state.my_status.get(id(state.my_active)) in (Status.PSN, Status.TOX):
            status_term -= 0.05

        # --- Active preservation ---
        my_active_role = expected_role_weight_for_mon(state.my_active, gen)
        if my_active_role > 1.06:
            active_preserve = _tanh01((my_active_hp - 0.60) / 0.20)
        else:
            active_preserve = _tanh01((my_active_hp - 0.45) / 0.25)

    # --- Tempo penalty (deeper is slightly worse) ---
    tempo_penalty = 0.04 * float(state.ply)

    # --- Progress / conversion ---
    # Encourage lines that actually reduce opponent resources, especially when ahead.
    opp_sum_now = _team_hp_sum(state.opp_hp)
    progress_term = _tanh01((1.0 - opp_sum_now) / 0.6)

    # --- Material lead (alive count) ---
    my_alive = sum(1 for v in state.my_hp.values() if float(v) > 0.0)
    opp_alive = sum(1 for v in state.opp_hp.values() if float(v) > 0.0)
    ahead = my_alive - opp_alive

    # --- Sac penalty when ahead ---
    sac_penalty = 0.0
    if ahead >= 2:
        if my_active_hp <= 0.0:
            sac_penalty += 0.25
        elif my_active_hp < 0.15:
            sac_penalty += 0.12
        elif my_active_hp < 0.30:
            sac_penalty += 0.06

    # --- Dynamic weights ---
    if ahead >= 2:
        w_team = 0.34
        w_numbers = 0.08
        w_race = 0.30
        w_switch = 0.05
        w_boost = 0.08
        w_active = 0.04
        w_progress = 0.15
    else:
        w_team = 0.38
        w_numbers = 0.10
        w_race = 0.25
        w_switch = 0.10
        w_boost = 0.10
        w_active = 0.07
        w_progress = 0.00

    value = (
        w_team * team_term +
        w_numbers * numbers_term +
        w_race * race_term +
        w_switch * switch_term +
        w_boost * boost_term +
        w_active * active_preserve +
        w_progress * progress_term +
        status_term
    ) - tempo_penalty - sac_penalty

    return max(-1.0, min(1.0, float(value)))
