import math
import json
import os
import re
from typing import Any, Optional

from poke_env.battle import Status, MoveCategory, Field, Weather, PokemonType
from bot.model.opponent_model import build_opponent_belief
from bot.scoring.race import evaluate_race_for_move

from bot.mcts.randbats_analyzer import (
    is_physical_attacker as rb_is_physical,
    is_fast_sweeper as rb_is_fast,
    is_defensive as rb_is_defensive,
    has_setup_potential as rb_has_setup,
    has_priority_moves as rb_has_priority,
)

def _tanh01(x: float) -> float:
    # maps to (-1, 1)
    return math.tanh(float(x))

def _safe_div(a: float, b: float, eps: float = 1e-9) -> float:
    return float(a / (b if abs(b) > eps else eps))

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

PIVOT_MOVE_IDS = {
    "uturn",
    "voltswitch",
    "flipturn",
    "partingshot",
    "teleport",
}

def _norm_move_id(mid: str) -> str:
    return str(mid).lower().replace("_", "").replace(" ", "")

def _has_pivot_move(mon: Any) -> bool:
    """Works with poke-env Pokemon objects that typically expose .moves (dict)."""
    moves = getattr(mon, "moves", None)
    if not moves:
        return False
    for mid in moves.keys():
        if _norm_move_id(mid) in PIVOT_MOVE_IDS:
            return True
    return False

def compute_switch_tax(side_conditions: dict, boots_prob: float = 0.0) -> float:
    if not side_conditions:
        return 0.0

    SR_TAX = 0.25
    SPIKE_PER_LAYER = 0.18
    TSP_1 = 0.12
    TSP_2 = 0.22
    WEB_TAX = 0.28

    b = max(0.0, min(1.0, float(boots_prob)))
    sr_mult = 1.0 - 0.70 * b
    sp_mult = 1.0 - 0.70 * b

    tax = 0.0
    if "stealthrock" in side_conditions:
        tax += SR_TAX * sr_mult

    spikes = int(side_conditions.get("spikes", 0) or 0)
    tax += (SPIKE_PER_LAYER * max(0, min(3, spikes))) * sp_mult

    tsp = int(side_conditions.get("toxicspikes", 0) or 0)
    if tsp == 1:
        tax += TSP_1
    elif tsp >= 2:
        tax += TSP_2

    if "stickyweb" in side_conditions:
        tax += WEB_TAX

    return tax

def compute_hazard_pressure(state: Any) -> float:
    my_sc = getattr(state, "my_side_conditions", {}) or {}
    opp_sc = getattr(state, "opp_side_conditions", {}) or {}

    opp_boots = avg_boots_prob_alive(state, "opp")     # belief-based since we do not know exactly what the opponent has
    my_boots = my_boots_frac_alive(state)              # exact

    opp_tax = compute_switch_tax(opp_sc, boots_prob=opp_boots)
    my_tax  = compute_switch_tax(my_sc,  boots_prob=my_boots)

    return float(opp_tax - my_tax)

def apply_hazard_scaling(race_term: float, switch_term: float, hazard_pressure: float) -> tuple[float, float]:
    """
    hazard_pressure is usually ~[-1, +1] with this tax model.
    Apply modest scaling to keep eval stable.
    """
    hp = max(-1.0, min(1.0, float(hazard_pressure)))
    tempo_scale = 1.0 + 0.22 * hp  # up to ±22%
    return (race_term * tempo_scale, switch_term * tempo_scale)

def compute_pivot_term(state: Any, hazard_pressure: float, uncertainty: float, race_term: float) -> float:
    me = getattr(state, "my_active", None)
    if me is None or not _has_pivot_move(me):
        return 0.0

    my_hp = float((getattr(state, "my_hp", {}) or {}).get(id(me), 0.0))

    # HP safety ramp: 0 at 20% HP, ~1 at 75% HP
    hp_safety = max(0.0, min(1.0, (my_hp - 0.20) / 0.55))

    # hazard amp: opponent-taxed => pivoting becomes better
    hp = max(-1.0, min(1.0, float(hazard_pressure)))
    hazard_amp = 1.0 + 0.25 * hp

    # race gate:
    # - if race_term << 0, pivot is often "too slow / too punished"
    # - if race_term >> 0, pivot is less necessary than just progressing
    r = max(-1.0, min(1.0, float(race_term)))
    if r < -0.50:
        race_gate = 0.25
    elif r < -0.20:
        race_gate = 0.55
    elif r > 0.50:
        race_gate = 0.70
    else:
        race_gate = 1.0

    raw = 0.12 * float(uncertainty) * hp_safety * hazard_amp * race_gate
    return float(max(-0.10, min(0.10, raw)))

def _boots_prob_for_belief(belief: Any) -> float:
    """
    Estimate P(Heavy-Duty Boots) from OpponentBelief.dist.
    Returns 0..1. Safe if belief is missing.
    """
    if belief is None:
        return 0.0
    dist = getattr(belief, "dist", None)
    if not dist:
        return 0.0
    total = 0.0
    for cand, p in dist:
        items = getattr(cand, "items", None) or set()
        if "heavydutyboots" in items:
            total += float(p)
    return max(0.0, min(1.0, total))


def avg_boots_prob_alive(state: Any, side: str) -> float:
    """
    Average P(boots) over alive mons for the given side.
    """
    if side != "opp":
        return 0.0

    beliefs = getattr(state, "opp_beliefs", {}) or {}
    opp_team = getattr(state, "opp_team", []) or []
    opp_hp = getattr(state, "opp_hp", {}) or {}

    probs = []
    for p in opp_team:
        if float(opp_hp.get(id(p), 0.0)) <= 0.0:
            continue
        probs.append(_boots_prob_for_belief(beliefs.get(id(p))))
    if not probs:
        return 0.0
    return float(sum(probs) / len(probs))

def my_boots_frac_alive(state: Any) -> float:
    my_team = getattr(state, "my_team", []) or []
    my_hp = getattr(state, "my_hp", {}) or {}
    alive = []
    for p in my_team:
        if float(my_hp.get(id(p), 0.0)) <= 0.0:
            continue
        alive.append(p)
    if not alive:
        return 0.0

    boots = 0
    for p in alive:
        item = getattr(p, "item", None)
        item_id = getattr(item, "id", None) or str(item) if item is not None else ""
        if str(item_id).lower().replace("_", "").replace(" ", "") == "heavydutyboots":
            boots += 1
    return boots / float(len(alive))

def belief_certainty(belief: Any) -> float:
    dist = getattr(belief, "dist", None)
    if not dist:
        return 0.0
    K = len(dist)
    if K <= 1:
        return 1.0
    H = 0.0
    for _, p in dist:
        p = float(p)
        if p > 1e-12:
            H -= p * math.log(p)
    Hmax = math.log(K)
    return float(max(0.0, min(1.0, 1.0 - (H / max(1e-12, Hmax)))))

def compute_info_terms(state: Any, opp_known: int, opp_total: int) -> tuple[float, float, float]:
    """
    Returns (roster_reveal, set_certainty, info_term) all in [0,1] (info_term is tanh-ish later).
    """
    roster_reveal = float(opp_known) / float(max(1, opp_total))

    beliefs = getattr(state, "opp_beliefs", {}) or {}
    opp_team = getattr(state, "opp_team", []) or []
    opp_hp = getattr(state, "opp_hp", {}) or {}

    certs = []
    for p in opp_team:
        if float(opp_hp.get(id(p), 0.0)) <= 0.0:
            continue
        b = beliefs.get(id(p))
        if b is None:
            continue
        certs.append(belief_certainty(b))

    set_certainty = float(sum(certs) / len(certs)) if certs else 0.0

    # Combine (roster is more important early than perfect set certainty)
    raw = 0.65 * roster_reveal + 0.35 * set_certainty
    info_term = float(max(0.0, min(1.0, raw)))
    return roster_reveal, set_certainty, info_term

def compute_belief_threat_term(state: Any) -> float:
    """
    Negative is bad for us (more likely sweep pressure).
    Kept modest so it doesn't overwhelm material/tempo.
    """
    beliefs = getattr(state, "opp_beliefs", {}) or {}
    opp_team = getattr(state, "opp_team", []) or []
    opp_hp = getattr(state, "opp_hp", {}) or {}
    opp_status = getattr(state, "opp_status", {}) or {}

    total = 0.0
    for p in opp_team:
        hp = float(opp_hp.get(id(p), 0.0))
        if hp <= 0.0:
            continue

        b = beliefs.get(id(p))
        if b is None or not getattr(b, "dist", None):
            continue

        # Expected tags under belief
        E_setup = 0.0
        E_prio = 0.0
        E_speed = 0.0
        E_phys = 0.0
        for cand, prob in b.dist:
            pr = float(prob)
            E_setup += pr * (1.0 if getattr(cand, "has_setup", False) else 0.0)
            E_prio  += pr * (1.0 if getattr(cand, "has_priority", False) else 0.0)
            E_speed += pr * float(getattr(cand, "speed_mult", 1.0))
            E_phys  += pr * float(getattr(cand, "physical_threat", 0.6))

        # Normalize speed_mult around 1.0 so it's a "threatiness" bump, not absolute speed.
        speed_excess = max(0.0, E_speed - 1.0)

        # Status discounts
        st = opp_status.get(id(p))
        # Para reduces speed-sweep pressure
        if st == Status.PAR:
            speed_excess *= 0.55
        # Burn reduces physical threat
        if st == Status.BRN:
            E_phys *= 0.60
        # Toxic/poison reduce setup-sweep pressure a bit (timer)
        if st in (Status.PSN, Status.TOX):
            E_setup *= 0.75

        # HP discount: low HP threats matter less
        hp_factor = max(0.35, min(1.0, hp / 0.80))

        threat = (
            0.70 * E_setup +
            0.55 * E_prio +
            0.60 * speed_excess +
            0.50 * E_phys
        ) * hp_factor

        total += threat

    # Map to a bounded negative term
    # (0 threat => 0, higher threat => more negative)
    return -_tanh01(total / 2.4)

def compute_setup_too_early_penalty(boost_term: float, uncertainty: float, opp_unseen: int) -> float:
    """
    Penalize relying on boosts when opponent is unknown (randbats counters / ditto risk).
    Applies mainly when boost_term is positive (we're trying to snowball).
    """
    if boost_term <= 0.0:
        return 0.0

    u = max(0.0, min(1.0, float(uncertainty)))
    # unseen amplifies risk slightly
    unseen_factor = max(0.0, min(1.0, opp_unseen / 4.0))

    # small penalty in [-0.10, 0]
    return -0.10 * boost_term * (0.6 * u + 0.4 * unseen_factor)

def compute_post_ko_danger_penalty(uncertainty: float, race_term: float, switch_term: float, my_active_hp: float, opp_unseen: int) -> float:
    """
    Penalize states where opponent can reveal a hidden threat for free.
    Strongest when: uncertainty high + you are slow/fragile + switches are poor.
    Returns negative.
    """
    u = max(0.0, min(1.0, float(uncertainty)))
    if u <= 0.05:
        return 0.0

    # Vulnerability signals
    # race_term < 0 => we're behind tempo
    tempo_bad = max(0.0, -float(race_term))
    # low HP => more exploitable
    hp_bad = max(0.0, (0.45 - float(my_active_hp)) / 0.45) 
    # if switch_term is low, we don't have good pivots/outs
    switch_bad = max(0.0, 0.6 - float(switch_term)) / 0.6 

    unseen = max(0, int(opp_unseen))
    unseen_amp = 0.6 + 0.4 * max(0.0, min(1.0, unseen / 3.0))

    vuln = (0.45 * tempo_bad + 0.35 * hp_bad + 0.20 * switch_bad) * unseen_amp
    return -0.10 * u * max(0.0, min(1.0, vuln))

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

def _move_id(m: Any) -> str:
    return _norm_id(getattr(m, "id", getattr(m, "name", "")))

def _mon_has_removal(mon: Any) -> bool:
    for mv in (getattr(mon, "moves", None) or {}).values():
        if mv is None:
            continue
        if _move_id(mv) in REMOVAL:
            return True
    return False

def _mon_has_damaging_priority(mon: Any) -> bool:
    # "speed control / priority user" ≈ has a damaging move with priority > 0
    for mv in (getattr(mon, "moves", None) or {}).values():
        if mv is None:
            continue
        pr = int(getattr(mv, "priority", 0) or 0)
        if pr <= 0:
            continue
        if getattr(mv, "category", None) == MoveCategory.STATUS:
            continue
        try:
            if int(getattr(mv, "base_power", 0) or 0) <= 0:
                continue
        except Exception:
            pass
        return True
    return False

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

def self_role_weight_for_mon(mon: Any, gen: int) -> float:
    """
    Lightweight role weight using *known* info only (our side).
    """
    w = 1.0

    # If we have setup / priority, higher leverage
    try:
        if has_setup_potential_perfect(mon):
            w *= 1.08
    except Exception:
        pass

    try:
        if has_priority_move_perfect(mon):
            w *= 1.04
    except Exception:
        pass

    # If we have hazards / removal, slightly higher strategic value
    try:
        for mv in (getattr(mon, "moves", None) or {}).values():
            if mv is None:
                continue
            mid = _move_id(mv)
            if mid in HAZARDS:
                w *= 1.04
                break
        for mv in (getattr(mon, "moves", None) or {}).values():
            if mv is None:
                continue
            if _move_id(mv) in REMOVAL:
                w *= 1.03
                break
    except Exception:
        pass

    # If we know we have a weather ability (rare on our side w/out reveal, but safe)
    try:
        abil = _norm_id(getattr(mon, "ability", "") or "")
        if abil in WEATHER_ABIL:
            w *= 1.06
    except Exception:
        pass

    return float(max(0.85, min(1.25, w)))


def opp_role_weight_from_belief(belief: Any) -> float:
    """
    Belief-averaged role weight for opponent mons.
    Uses persistent belief (state.opp_beliefs)
    """
    try:
        if belief is None or not getattr(belief, "dist", None):
            return 1.0
        w = 0.0
        for c, p in belief.dist:
            w += float(p) * float(candidate_role_weight(c))
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

def team_value(team, hp_map, boosts_map, gen: int, status_map=None, *, side: str = "me", opp_beliefs: Optional[dict] = None) -> float:
    total = 0.0
    status_map = status_map or {}
    opp_beliefs = opp_beliefs or {}

    alive = [m for m in team if float(hp_map.get(id(m), 0.0)) > 0.0]
    num_removers = sum(1 for m in alive if _mon_has_removal(m))
    num_priority = sum(1 for m in alive if _mon_has_damaging_priority(m))

    for mon in team:
        hp = float(hp_map.get(id(mon), 0.0))
        if hp <= 0.0:
            continue

        # Role weight: self vs opponent (belief-based)
        if side == "opp":
            role_w = opp_role_weight_from_belief(opp_beliefs.get(id(mon)))
        else:
            role_w = self_role_weight_for_mon(mon, gen)

        boosts = (boosts_map.get(id(mon), {}) if boosts_map else {}) or {}
        st = status_map.get(id(mon), None)

        unique_mult = 1.0
        try:
            if num_removers == 1 and _mon_has_removal(mon):
                unique_mult *= 1.10
            if num_priority == 1 and _mon_has_damaging_priority(mon):
                unique_mult *= 1.07
        except Exception:
            pass

        v = hp
        v *= role_w
        v *= unique_mult
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


def _physical_ratio(team, hp_map) -> float:
    """Fraction of alive team that is physical-leaning. Returns 0.0-1.0."""
    phys = 0
    alive = 0
    for mon in team:
        if float(hp_map.get(id(mon), 0.0)) <= 0.0:
            continue
        alive += 1
        if is_physical_attacker_perfect(mon):
            phys += 1
    if alive == 0:
        return 0.5
    return phys / alive


def _screen_survival_bonus(state: Any, screen: str, side: str, turns_left: int) -> float:
    """
    Extra value when a screen changes KO thresholds for the active matchup.
    Scales with remaining turns — a screen about to expire matters less.
    Returns 0.0 ~ 0.08 bonus.
    """
    if side == "me":
        defender = state.my_active
        defender_hp = float(state.my_hp.get(id(defender), 0.0))
        attacker = state.opp_active
    else:
        defender = state.opp_active
        defender_hp = float(state.opp_hp.get(id(defender), 0.0))
        attacker = state.my_active

    if defender_hp <= 0.0:
        return 0.0

    # Check if the screen's category matches the opponent's best threat
    if screen == 'reflect':
        if not is_physical_attacker_perfect(attacker):
            return 0.0
    elif screen == 'lightscreen':
        if is_physical_attacker_perfect(attacker):
            return 0.0

    # Base bonus by defender HP
    if defender_hp < 0.30:
        base = 0.02  # screen helps less when already low
    elif defender_hp < 0.60:
        base = 0.05  # decent value, buys a turn
    else:
        base = 0.03  # healthy, screen is insurance

    # Scale by remaining turns (out of max 5): 1 turn left → 20%, 5 turns → 100%
    turn_factor = min(1.0, turns_left / 5.0)
    return base * turn_factor


def _weather_affinity(team, hp_map, weather) -> float:
    """How much a team benefits from the current weather. Returns -1.0 to +1.0."""
    score = 0.0
    alive = 0
    for mon in team:
        if float(hp_map.get(id(mon), 0.0)) <= 0.0:
            continue
        alive += 1
        types = getattr(mon, 'types', [])
        if weather in (Weather.SUNNYDAY,):
            if PokemonType.FIRE in types:
                score += 1.0   # boosted STAB
            if PokemonType.WATER in types:
                score -= 0.5   # weakened STAB
            if PokemonType.GRASS in types:
                score += 0.3   # no solar beam charge
        elif weather in (Weather.RAINDANCE,):
            if PokemonType.WATER in types:
                score += 1.0
            if PokemonType.FIRE in types:
                score -= 0.5
        elif weather in (Weather.SANDSTORM,):
            if PokemonType.ROCK in types:
                score += 0.6   # SpD boost
            if PokemonType.STEEL in types or PokemonType.GROUND in types:
                score += 0.3   # immune to chip
        elif weather in (Weather.SNOW, Weather.HAIL):
            if PokemonType.ICE in types:
                score += 0.6   # Def boost in snow
    if alive == 0:
        return 0.0
    return score / alive


def _status_value_for_mon(mon: Any, status, volatiles: dict, is_active: bool) -> float:
    """
    How bad a status condition is for a specific Pokemon.
    Returns a positive value (higher = worse for the afflicted mon).
    """
    value = 0.0

    # Primary statuses 
    if status == Status.BRN:
        if is_physical_attacker_perfect(mon):
            value += 0.18  # halved attack on a physical attacker
        else:
            value += 0.08  # chip damage only for special attackers
    elif status == Status.PAR:
        if is_fast_sweeper_perfect(mon):
            value += 0.14  # speed cut on a fast sweeper is devastating
        else:
            value += 0.06  # 25% full para chance still hurts
    elif status == Status.TOX:
        if is_defensive_perfect(mon):
            value += 0.14  # toxic on a wall is very strong
        else:
            value += 0.08  # escalating chip
    elif status == Status.PSN:
        if is_defensive_perfect(mon):
            value += 0.10  # steady chip on a wall
        else:
            value += 0.05
    elif status == Status.SLP:
        if is_active:
            sleep_turns = volatiles.get('sleep_turns', 0)
            if sleep_turns >= 2:
                value += 0.10  # likely waking soon
            else:
                value += 0.22  # multiple turns of denial — very punishing
        else:
            value += 0.15  # benched sleeper is still bad (can't switch in safely)
    elif status == Status.FRZ:
        if is_active:
            value += 0.20  # frozen = complete denial (20% thaw per turn)
        else:
            value += 0.15

    # Volatile statuses (active only) 
    if is_active:
        conf_turns = volatiles.get('confusion_turns', 0)
        if conf_turns > 0:
            if conf_turns >= 3:
                value += 0.04  # likely ending soon
            else:
                value += 0.08  # ~33% self-hit chance per turn

    return value


def evaluate_status_conditions(state: Any) -> float:
    """
    Evaluate status + volatile conditions for both sides.
    Positive = good for us (opponent more afflicted), negative = bad.
    """
    my_cost = 0.0
    opp_cost = 0.0

    # Active Pokemon statuses + volatiles
    my_active_id = id(state.my_active)
    opp_active_id = id(state.opp_active)

    my_active_st = state.my_status.get(my_active_id)
    my_active_vol = state.my_volatiles.get(my_active_id, {})
    my_cost += _status_value_for_mon(state.my_active, my_active_st, my_active_vol, is_active=True)

    opp_active_st = state.opp_status.get(opp_active_id)
    opp_active_vol = state.opp_volatiles.get(opp_active_id, {})
    opp_cost += _status_value_for_mon(state.opp_active, opp_active_st, opp_active_vol, is_active=True)

    # Bench Pokemon statuses (no volatiles — cleared on switch)
    for mon in state.my_team:
        if mon is state.my_active:
            continue
        if float(state.my_hp.get(id(mon), 0.0)) <= 0.0:
            continue
        st = state.my_status.get(id(mon))
        if st is not None:
            my_cost += _status_value_for_mon(mon, st, {}, is_active=False)

    for mon in state.opp_team:
        if mon is state.opp_active:
            continue
        if float(state.opp_hp.get(id(mon), 0.0)) <= 0.0:
            continue
        st = state.opp_status.get(id(mon))
        if st is not None:
            opp_cost += _status_value_for_mon(mon, st, {}, is_active=False)

    # Net: opponent being statused is good for us
    return opp_cost - my_cost

def evaluate_field_conditions(state: Any) -> float:
    """Evaluate strategic value of field conditions beyond damage modifiers."""
    value = 0.0
    my_sc = getattr(state, 'my_side_conditions', {}) or {}
    opp_sc = getattr(state, 'opp_side_conditions', {}) or {}
    fields = getattr(state, 'shadow_fields', {}) or {}

    # Opponent attack profile determines how much each screen matters
    opp_phys = _physical_ratio(state.opp_team, state.opp_hp)
    my_phys = _physical_ratio(state.my_team, state.my_hp)

    # Reflect: valuable proportional to opponent's physical ratio
    # Light Screen: valuable proportional to opponent's special ratio
    reflect_weight = 0.015 + 0.015 * opp_phys
    lscreen_weight = 0.015 + 0.015 * (1 - opp_phys)

    # Our screens vs their screens (per remaining turn)
    value += reflect_weight * my_sc.get('reflect', 0)
    value += lscreen_weight * my_sc.get('lightscreen', 0)

    # Their screens against us (mirrored)
    opp_reflect_weight = 0.015 + 0.015 * my_phys
    opp_lscreen_weight = 0.015 + 0.015 * (1 - my_phys)
    value -= opp_reflect_weight * opp_sc.get('reflect', 0)
    value -= opp_lscreen_weight * opp_sc.get('lightscreen', 0)

    # Survival bonus: extra value when screen changes KO threshold, scaled by turns left
    for scr in ('reflect', 'lightscreen'):
        my_turns = my_sc.get(scr, 0)
        opp_turns = opp_sc.get(scr, 0)
        if my_turns > 0:
            value += _screen_survival_bonus(state, scr, 'me', my_turns)
        if opp_turns > 0:
            value -= _screen_survival_bonus(state, scr, 'opp', opp_turns)

    # Aurora Veil (both physical+special, always valuable)
    value += 0.025 * my_sc.get('auroraveil', 0)
    value -= 0.025 * opp_sc.get('auroraveil', 0)

    # Tailwind (speed doubling = tempo advantage)
    value += 0.03 * my_sc.get('tailwind', 0)
    value -= 0.03 * opp_sc.get('tailwind', 0)

    # Trick Room (benefits slow teams, scaled by remaining turns)
    if Field.TRICK_ROOM in fields:
        tr_counter = fields[Field.TRICK_ROOM]
        tr_remaining = max(0, 5 - tr_counter)
        tr_factor = tr_remaining / 5.0
        my_spe = state._effective_speed(state.my_active, "me")
        opp_spe = state._effective_speed(state.opp_active, "opp")
        if my_spe < opp_spe:
            value += 0.08 * tr_factor
        elif opp_spe < my_spe:
            value -= 0.08 * tr_factor

    # Weather advantage (scaled by remaining turns)
    weather = getattr(state, 'shadow_weather', {}) or {}
    for w, counter in weather.items():
        w_remaining = max(0, 5 - counter)
        w_factor = w_remaining / 5.0
        my_affinity = _weather_affinity(state.my_team, state.my_hp, w)
        opp_affinity = _weather_affinity(state.opp_team, state.opp_hp, w)
        value += 0.06 * (my_affinity - opp_affinity) * w_factor

    # Hazards the opponent cannot remove are stable pressure
    my_has_removal = any(
        _mon_has_removal(p) for p in state.my_team
        if float(state.my_hp.get(id(p), 0.0)) > 0.0
    )
    opp_has_removal = opp_removal_prob_alive(state) # Using belief system here
    if any(h in opp_sc for h in HAZARDS) and not opp_has_removal:
        value += 0.08 * (1.0 - opp_has_removal)
    if any(h in my_sc for h in HAZARDS) and not my_has_removal:
        value -= 0.08  # their hazards on our side are permanent

    # Toxic Spikes are more impactful vs grounded, non-immune mons
    _TSP_IMMUNE = {PokemonType.POISON, PokemonType.STEEL, PokemonType.FLYING}
    tsp_opp = opp_sc.get('toxicspikes', 0)
    if tsp_opp >= 1:
        opp_alive = [p for p in state.opp_team if float(state.opp_hp.get(id(p), 0.0)) > 0.0]
        if opp_alive:
            vulnerable = sum(
                1 for p in opp_alive
                if not any(t in getattr(p, 'types', []) for t in _TSP_IMMUNE)
            )
            value += (vulnerable / len(opp_alive)) * min(tsp_opp, 2) * 0.04

    tsp_my = my_sc.get('toxicspikes', 0)
    if tsp_my >= 1:
        my_alive = [p for p in state.my_team if float(state.my_hp.get(id(p), 0.0)) > 0.0]
        if my_alive:
            vulnerable = sum(
                1 for p in my_alive
                if not any(t in getattr(p, 'types', []) for t in _TSP_IMMUNE)
            )
            value -= (vulnerable / len(my_alive)) * min(tsp_my, 2) * 0.04

    return _tanh01(value / 0.3)

def is_physical_attacker_perfect(mon: Any) -> bool:
    """
    Check if Pokemon is physical attacker using randbats database.
    Falls back to base stats if needed.
    """
    try:
        return rb_is_physical(mon)
    except:
        # Fallback to base stats
        try:
            atk = mon.base_stats.get('atk', 100)
            spa = mon.base_stats.get('spa', 100)
            return atk > spa * 1.1
        except:
            return True


def is_fast_sweeper_perfect(mon: Any) -> bool:
    """
    Check if Pokemon is fast sweeper using randbats database.
    Falls back to base speed if needed.
    """
    try:
        return rb_is_fast(mon)
    except:
        # Fallback to base speed
        try:
            spe = mon.base_stats.get('spe', 100)
            return spe >= 100
        except:
            return False


def is_defensive_perfect(mon: Any) -> bool:
    """
    Check if Pokemon is defensive using randbats database.
    Falls back to bulk calculation if needed.
    """
    try:
        return rb_is_defensive(mon)
    except:
        # Fallback to bulk calculation
        try:
            hp = mon.base_stats.get('hp', 100)
            defense = mon.base_stats.get('def', 100)
            spdef = mon.base_stats.get('spd', 100)
            bulk = hp * (defense + spdef) / 2
            return bulk > 15000
        except:
            return False


def has_setup_potential_perfect(mon: Any) -> bool:
    """
    Check if Pokemon has setup moves using randbats database.
    Falls back to checking known moves if needed.
    """
    try:
        return rb_has_setup(mon)
    except:
        # Fallback: check known moves
        for move in (getattr(mon, 'moves', None) or {}).values():
            move_id = _norm_id(str(getattr(move, 'id', '') or ''))
            if move_id in SETUP_MOVES:
                return True
        return False


def has_priority_move_perfect(mon: Any) -> bool:
    """
    Check if Pokemon has priority moves using randbats database.
    Falls back to checking known moves if needed.
    """
    try:
        return rb_has_priority(mon)
    except:
        # Fallback: check known moves
        for move in (getattr(mon, 'moves', None) or {}).values():
            if move is None:
                continue
            priority = int(getattr(move, 'priority', 0) or 0)
            if priority > 0 and getattr(move, 'base_power', 0) > 0:
                return True
        return False
    
def _removal_prob_for_belief(belief: Any) -> float:
    if belief is None or not getattr(belief, "dist", None):
        return 0.0
    total = 0.0
    for cand, p in belief.dist:
        # cand.moves is a set of normalized ids in your candidate objects
        if any(m in REMOVAL for m in getattr(cand, "moves", set()) or set()):
            total += float(p)
    return max(0.0, min(1.0, total))

def opp_removal_prob_alive(state: Any) -> float:
    beliefs = getattr(state, "opp_beliefs", {}) or {}
    opp_team = getattr(state, "opp_team", []) or []
    opp_hp = getattr(state, "opp_hp", {}) or {}

    probs = []
    for p in opp_team:
        if float(opp_hp.get(id(p), 0.0)) <= 0.0:
            continue
        probs.append(_removal_prob_for_belief(beliefs.get(id(p))))
    return float(sum(probs) / len(probs)) if probs else 0.0

def _sack_bench_quality(state: Any, gen: int) -> float:
    active_role = self_role_weight_for_mon(state.my_active, gen)
    dispensable_factor = max(0.0, min(1.0, (1.10 - active_role) / 0.10))  # 0..1

    bench = [
        p for p in state.my_team
        if p is not state.my_active and float(state.my_hp.get(id(p), 0.0)) > 0.30
    ]
    if not bench:
        return 0.0

    best_hp = max(float(state.my_hp.get(id(p), 0.0)) for p in bench)

    setup_factor = 1.0
    for p in bench:
        if has_setup_potential_perfect(p) and float(state.my_hp.get(id(p), 0.0)) > 0.60:
            setup_factor = 1.5
            break

    return min(1.0, best_hp * dispensable_factor * setup_factor)

def evaluate_sack_opportunity(state: Any, gen: int) -> float:
    my_active_hp = float(state.my_hp.get(id(state.my_active), 0.0))
    opp_active_hp = float(state.opp_hp.get(id(state.opp_active), 0.0))

    if my_active_hp <= 0.0 or opp_active_hp <= 0.0:
        return 0.0
    if my_active_hp > 0.50:
        return 0.0

    bench = [
        p for p in state.my_team
        if p is not state.my_active and float(state.my_hp.get(id(p), 0.0)) > 0.0
    ]
    if not bench:
        return 0.0

    # Important mons should never be sacked
    active_role = self_role_weight_for_mon(state.my_active, gen)
    if active_role > 1.10:
        return 0.0
    dispensable_factor = max(0.0, (1.10 - active_role) / 0.10)  # 0..1

    if state.score_switch_fn is None or state.ctx_me is None:
        return 0.0
    best_switch_score = max(
        float(state.score_switch_fn(p, state.battle, state.ctx_me))
        for p in bench
    )
    SW_NORM = 2.0
    bench_advantage = max(0.0, _tanh01(best_switch_score / SW_NORM))

    if bench_advantage < 0.15:
        return 0.0

    setup_bonus = 1.0
    for p in bench:
        if has_setup_potential_perfect(p) and float(state.my_hp.get(id(p), 0.0)) > 0.50:
            setup_bonus = 1.35
            break

    danger_factor = max(0.0, 0.50 - my_active_hp) / 0.50

    return 0.12 * bench_advantage * dispensable_factor * danger_factor * setup_bonus

def evaluate_state(state: Any) -> float:
    """
    Returns a scalar value for MCTS backup: higher is better for us.
    Range ~[-1, +1].
    """
    my_sum_raw = _team_hp_sum(state.my_hp)
    opp_sum_raw = _team_hp_sum(state.opp_hp)

    if my_sum_raw <= 1e-9:
        return -1.0

    opp_known = len(getattr(state, "opp_team", []) or [])
    opp_total = int(getattr(state, "opp_team_size", 6) or 6)
    battle_finished = bool(getattr(state.battle, "finished", False))

    if opp_sum_raw <= 1e-9 and (battle_finished or opp_known >= opp_total):
        return +1.0

    my_active_hp = float(state.my_hp.get(id(state.my_active), 0.0))
    opp_active_hp = float(state.opp_hp.get(id(state.opp_active), 0.0))

    try:
        gen = int(getattr(state.battle, "gen", 9) or 9)
    except (TypeError, ValueError):
        gen = 9

    # If our active is fainted but theirs isn't, clamp negative (with strategic-sack softness)
    if my_active_hp <= 0.0 and opp_active_hp > 0.0:
        lead_hint = _tanh01((my_sum_raw - opp_sum_raw) / 1.5)
        base = -0.90 + 0.15 * lead_hint
        bench_qual = _sack_bench_quality(state, gen)
        base += 0.35 * bench_qual
        return max(-1.0, min(1.0, float(base)))

    # Endgame detection
    my_alive_count = sum(1 for v in state.my_hp.values() if v > 0)
    opp_alive_count = sum(1 for v in state.opp_hp.values() if v > 0)

    if my_alive_count == 1 and opp_alive_count == 1:
        endgame_value = 0.0
        hp_diff = my_active_hp - opp_active_hp
        endgame_value += _tanh01(hp_diff / 0.4)

        if has_priority_move_perfect(state.my_active) and not has_priority_move_perfect(state.opp_active):
            endgame_value += 0.10
        elif has_priority_move_perfect(state.opp_active) and not has_priority_move_perfect(state.my_active):
            endgame_value -= 0.10

        my_status_eg = state.my_status.get(id(state.my_active))
        opp_status_eg = state.opp_status.get(id(state.opp_active))
        if my_status_eg in (Status.PSN, Status.TOX) and opp_status_eg not in (Status.PSN, Status.TOX):
            endgame_value -= 0.12
        elif opp_status_eg in (Status.PSN, Status.TOX) and my_status_eg not in (Status.PSN, Status.TOX):
            endgame_value += 0.12

        return max(-1.0, min(1.0, endgame_value))

    elif my_alive_count == 1 and opp_alive_count >= 2:
        my_boosts_eg = state.my_boosts.get(id(state.my_active), {})
        max_boost = max((v for v in my_boosts_eg.values() if v > 0), default=0)

        if has_setup_potential_perfect(state.my_active) and my_active_hp > 0.7 and max_boost < 2:
            return -0.30
        elif max_boost >= 4:
            return -0.10
        else:
            return -0.70

    elif my_alive_count >= 2 and opp_alive_count == 1:
        return +0.70

    with state._patched_status(), state._patched_boosts(), state._patched_fields():
        my_value = team_value(
            state.my_team, state.my_hp, state.my_boosts, gen,
            status_map=state.my_status,
            side="me",
        )

        opp_value_known = team_value(
            state.opp_team, state.opp_hp, state.opp_boosts, gen,
            status_map=state.opp_status,
            side="opp",
            opp_beliefs=(getattr(state, "opp_beliefs", {}) or {}),
        )

        opp_unseen = max(0, opp_total - opp_known)
        opp_value = opp_value_known + opp_unseen_value(opp_known, opp_total)
        team_term = _tanh01((my_value - opp_value) / 1.2)

        my_healthy = healthy_count(state.my_team, state.my_hp, 0.55)
        opp_healthy = healthy_count(state.opp_team, state.opp_hp, 0.55) + opp_unseen
        numbers_term = _tanh01((my_healthy - opp_healthy) / 1.5)

        # Best-move race
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
            my_eff_speed = state._effective_speed(state.my_active, "me")
            opp_eff_speed = state._effective_speed(state.opp_active, "opp")
            if Field.TRICK_ROOM in getattr(state, 'shadow_fields', {}):
                my_eff_speed, opp_eff_speed = opp_eff_speed, my_eff_speed
            race = evaluate_race_for_move(
                state.battle, state.ctx_me, best_mv,
                me_speed=my_eff_speed, opp_speed=opp_eff_speed,
            )
            race_term = _tanh01((race.ttd_me - race.tko_opp) / 1.5)

        # Always evaluate best bench option; weight is gated in the sum below
        best_sw_score = -1e18
        for p in state.my_team:
            if p is state.my_active:
                continue
            if float(state.my_hp.get(id(p), 0.0)) <= 0.0:
                continue
            sc = float(state.score_switch_fn(p, state.battle, state.ctx_me))
            if sc > best_sw_score:
                best_sw_score = sc
        SW_NORM = 2.0
        switch_term = _tanh01(best_sw_score / SW_NORM) if best_sw_score > -1e17 else 0.0

        # Hazards => tempo scaling
        hazard_pressure = compute_hazard_pressure(state)
        race_term, switch_term = apply_hazard_scaling(race_term, switch_term, hazard_pressure)

        # Boost term
        boost_term = evaluate_boosts(state)
        if my_active_hp < 0.20:
            boost_term *= 0.40
        elif my_active_hp < 0.35:
            boost_term *= 0.70

        # Info terms (used for gating only)
        roster_reveal, set_certainty, info_term01 = compute_info_terms(state, opp_known, opp_total)
        uncertainty = 1.0 - info_term01

        # Pivot term
        pivot_term = compute_pivot_term(state, hazard_pressure, uncertainty, race_term)

        # Threat term
        threat_term = compute_belief_threat_term(state)

        # Setup-too-early penalty
        setup_early_pen = compute_setup_too_early_penalty(boost_term, uncertainty, opp_unseen)

        # Post-KO danger penalty (fix switch_bad calibration via switch_good01)
        post_ko_pen = compute_post_ko_danger_penalty(
            uncertainty=uncertainty,
            race_term=race_term,
            switch_term=0.5 * (switch_term + 1.0),  # <-- IMPORTANT: map to [0,1] goodness
            my_active_hp=my_active_hp,
            opp_unseen=opp_unseen,
        )

        field_term = evaluate_field_conditions(state)

        status_term = evaluate_status_conditions(state)
        status_term = max(-1.0, min(1.0, float(status_term)))  # <-- clamp

        sack_bonus = evaluate_sack_opportunity(state, gen)

        # Active preservation: avoid using opponent-belief builder for our own mon
        # For now: treat as slightly more important if it has setup/priority
        if has_setup_potential_perfect(state.my_active) or has_priority_move_perfect(state.my_active):
            active_preserve = _tanh01((my_active_hp - 0.60) / 0.20)
        else:
            active_preserve = _tanh01((my_active_hp - 0.45) / 0.25)

    tempo_penalty = 0.04 * float(state.ply)

    opp_sum_now = _team_hp_sum(state.opp_hp)
    progress_term = _tanh01((1.0 - opp_sum_now) / 0.6)

    my_alive = sum(1 for v in state.my_hp.values() if float(v) > 0.0)
    opp_alive = sum(1 for v in state.opp_hp.values() if float(v) > 0.0)
    ahead = my_alive - opp_alive

    ahead_factor = max(0.0, min(1.0, (ahead - 1) / 3.0))
    active_preserve *= (1.0 - 0.50 * ahead_factor)

    sac_penalty = 0.0
    if ahead >= 2:
        if my_active_hp <= 0.0:
            sac_penalty += 0.02 if opp_active_hp <= 0.0 else 0.20
        elif my_active_hp < 0.15:
            sac_penalty += 0.10
        elif my_active_hp < 0.30:
            sac_penalty += 0.05
    sac_penalty = max(0.0, sac_penalty - sack_bonus)

    # Weights (drop w_info entirely)
    if ahead >= 2:
        w_team     = 0.28
        w_numbers  = 0.06
        w_race     = 0.28
        w_switch   = 0.05
        w_boost    = 0.07
        w_active   = 0.04
        w_progress = 0.14
        w_field    = 0.05
        w_pivot    = 0.03
        w_threat   = 0.03
        w_status   = 0.06
    else:
        w_team     = 0.32
        w_numbers  = 0.07
        w_race     = 0.22
        w_switch   = 0.09
        w_boost    = 0.09
        w_active   = 0.06
        w_progress = 0.00
        w_field    = 0.07
        w_pivot    = 0.05
        w_threat   = 0.04
        w_status   = 0.07

    # Switch fully weighted when losing race, reduced when winning
    effective_w_switch = w_switch if race_term < 0.0 else w_switch * 0.30

    # Core weighted sum 
    core = (
        w_team * team_term +
        w_numbers * numbers_term +
        w_race * race_term +
        effective_w_switch * switch_term +
        w_boost * boost_term +
        w_active * active_preserve +
        w_progress * progress_term +
        w_field * field_term +
        w_status * status_term +
        w_pivot * pivot_term +
        w_threat * threat_term
    )

    w_sum = (w_team + w_numbers + w_race + effective_w_switch + w_boost + w_active +
            w_progress + w_field + w_status + w_pivot + w_threat)

    core_norm = _safe_div(core, w_sum)

    # Apply penalties/bonuses after normalization (keeps them interpretable)
    value = core_norm - tempo_penalty - sac_penalty + setup_early_pen + post_ko_pen

    return max(-1.0, min(1.0, float(value)))