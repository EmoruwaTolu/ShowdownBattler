"""
Structured state encoding for the battle value model.

Returns three arrays from a ShadowState:

  tokens     : float32 (12, N_MON_FEATURES)      per-Pokemon tokens
  field      : float32 (N_FIELD_FEATURES,)        weather/terrain/hazards + team summary
  active_pair: float32 (N_ACTIVE_PAIR_FEATURES,)  active-vs-active tactical state

Flat MLP input (N_TOTAL,) = concat(tokens.reshape(-1), field, active_pair).

Slot ordering
  Slots 0-5  : your team  — active first, bench alive, fainted last
  Slots 6-11 : opponent   — active first, bench alive/revealed, fainted/unseen last

Unseen opponent sentinel: hp_frac=1.0, belief_entropy=1.0, all else 0.

All entry points are safe to call during MCTS rollouts; they never raise.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional poke_env imports — fail gracefully in unit tests
# ---------------------------------------------------------------------------
try:
    from poke_env.battle import Status
    _HAS_STATUS = True
except ImportError:
    Status = None  # type: ignore
    _HAS_STATUS = False

try:
    from poke_env.battle import Effect
    _HAS_EFFECT = True
except ImportError:
    Effect = None  # type: ignore
    _HAS_EFFECT = False

try:
    from poke_env.data import GenData
    _TYPE_CHART: Dict[str, Dict[str, float]] = GenData.from_gen(9).type_chart
except Exception:
    _TYPE_CHART = {}

try:
    from bot.scoring.damage_score import estimate_damage_fraction as _estimate_dmg
except ImportError:
    _estimate_dmg = None  # type: ignore

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

N_MON_FEATURES        = 32
N_FIELD_FEATURES      = 23
N_ACTIVE_PAIR_FEATURES = 3
N_TOTAL = 12 * N_MON_FEATURES + N_FIELD_FEATURES + N_ACTIVE_PAIR_FEATURES  # 410

MON_FEATURE_NAMES: List[str] = [
    # Identity
    "hp_frac",           # 0  [0,1]
    "is_active",         # 1  {0,1}
    "is_fainted",        # 2  {0,1}
    # Status
    "status_burn",       # 3  {0,1}
    "status_poison",     # 4  {0,1}
    "status_toxic",      # 5  {0,1}
    "status_paralysis",  # 6  {0,1}
    "status_sleep",      # 7  {0,1}
    "status_freeze",     # 8  {0,1}
    # Stat boosts — stage / 6, range [-1, 1]
    "atk_boost",         # 9
    "spa_boost",         # 10
    "spe_boost",         # 11
    "def_boost",         # 12
    "spd_boost",         # 13
    # Matchup vs active opponent
    "type_adv",          # 14 [0,1]  max eff this mon → opp / 4
    "type_disadv",       # 15 [0,1]  max eff opp → this mon / 4
    "best_dmg_frac",     # 16 [0,1]  best move damage / opp max HP
    # Move roles
    "has_setup",         # 17 {0,1}
    "has_recovery",      # 18 {0,1}
    "has_priority",      # 19 {0,1}
    "has_pivot",         # 20 {0,1}
    "has_hazard_clear",  # 21 {0,1}
    # Information quality
    "move_count_known",  # 22 [0,1]  confirmed moves / 4
    "belief_entropy",    # 23 [0,1]  normalized entropy over possible sets
    # Volatile conditions (active-pokemon state; bench slots get 0)
    "has_substitute",    # 24 {0,1}
    "has_leech_seed",    # 25 {0,1}
    "is_taunted",        # 26 {0,1}
    "is_encored",        # 27 {0,1}
    "is_confused",       # 28 {0,1}
    "perish_turns",      # 29 [0,1]  perish counter / 3 (0 = no perish)
    "yawn_incoming",     # 30 {0,1}  will fall asleep next turn
    "is_terastallized",  # 31 {0,1}
]

FIELD_FEATURE_NAMES: List[str] = [
    # Weather
    "weather_sun",         # 0   {0,1}
    "weather_rain",        # 1   {0,1}
    "weather_sand",        # 2   {0,1}
    "weather_snow",        # 3   {0,1}
    # Terrain
    "terrain_electric",    # 4   {0,1}
    "terrain_grassy",      # 5   {0,1}
    "terrain_psychic",     # 6   {0,1}
    "terrain_misty",       # 7   {0,1}
    # My side conditions
    "my_stealth_rock",     # 8   {0,1}
    "my_spikes",           # 9   [0,1]  layers / 3
    "my_toxic_spikes",     # 10  [0,1]  layers / 2
    "my_sticky_web",       # 11  {0,1}
    # Opponent side conditions
    "opp_stealth_rock",    # 12  {0,1}
    "opp_spikes",          # 13  [0,1]
    "opp_toxic_spikes",    # 14  [0,1]
    "opp_sticky_web",      # 15  {0,1}
    # Team-level summary signals
    "moves_first",         # 16  {-1,0,1}  +1=you, -1=opp, 0=tie/unknown
    "team_hp_diff",        # 17  [-1,1]
    "alive_count_diff",    # 18  [-1,1]
    "best_switch_safety",  # 19  [0,1]
    "hazard_pressure",     # 20  [-1,1]  net future switch cost in your favor
    # Tera usage (one Tera per game per side)
    "my_tera_used",        # 21  {0,1}  we have already Terastallized this game
    "opp_tera_used",       # 22  {0,1}  opponent has already Terastallized this game
]

ACTIVE_PAIR_FEATURE_NAMES: List[str] = [
    "can_ko",               # 0  {0,1}  your active can KO opponent active this turn
    "can_be_ko",            # 1  {0,1}  opponent active can KO your active this turn
    "speed_tie_or_unknown", # 2  {0,1}  1 if move order is not deterministic
]

assert len(MON_FEATURE_NAMES)         == N_MON_FEATURES
assert len(FIELD_FEATURE_NAMES)       == N_FIELD_FEATURES
assert len(ACTIVE_PAIR_FEATURE_NAMES) == N_ACTIVE_PAIR_FEATURES

# ---------------------------------------------------------------------------
# Move-role id sets  (keep in sync with action_features.py)
# ---------------------------------------------------------------------------

def _norm_id(s: Any) -> str:
    base = getattr(s, "id", None) or getattr(s, "name", None) or str(s)
    return str(base).lower().replace("_", "").replace("-", "").replace(" ", "")


_SETUP_IDS = frozenset({
    "swordsdance", "nastyplot", "dragondance", "calmmind", "bulkup",
    "quiverdance", "shellsmash", "bellydrum", "shiftgear", "agility",
    "tailglow", "coil", "curse", "growth", "workup", "honeclaws",
    "cottonguard", "irondefense", "acidarmor",
})
_RECOVERY_IDS = frozenset({
    "recover", "softboiled", "slackoff", "roost", "moonlight", "morningsun",
    "synthesis", "shoreup", "leechseed", "strengthsap", "oblivionwing",
    "healorder", "milkdrink", "lifedew", "lunarblessing",
})
_PIVOT_IDS = frozenset({
    "uturn", "voltswitch", "flipturn", "partingshot", "chillyreception", "teleport",
})
_HAZARD_CLEAR_IDS = frozenset({"rapidspin", "defog"})

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _type_effectiveness(atk_type: Any, defender: Any) -> float:
    try:
        def_types = [t for t in (getattr(defender, "types", None) or []) if t is not None]
        if not def_types or not _TYPE_CHART:
            return 1.0
        atk_name = getattr(atk_type, "name", str(atk_type)).lower()
        mult = 1.0
        for dt in def_types:
            dt_name = getattr(dt, "name", str(dt)).lower()
            mult *= float(_TYPE_CHART.get(atk_name, {}).get(dt_name, 1.0))
        return mult
    except Exception:
        return 1.0


def _max_type_adv(attacker: Any, defender: Any) -> float:
    """Best type effectiveness attacker → defender, normalized [0,1] (4× = 1.0)."""
    try:
        atk_types = [t for t in (getattr(attacker, "types", None) or []) if t is not None]
        if not atk_types:
            return 0.25
        best = max(_type_effectiveness(t, defender) for t in atk_types)
        return min(1.0, best / 4.0)
    except Exception:
        return 0.25


def _status_vec(status: Any) -> List[float]:
    """[burn, poison, toxic, paralysis, sleep, freeze]."""
    if status is None or not _HAS_STATUS:
        return [0.0] * 6
    try:
        return [
            1.0 if status == Status.BRN else 0.0,
            1.0 if status == Status.PSN else 0.0,
            1.0 if status == Status.TOX else 0.0,
            1.0 if status == Status.PAR else 0.0,
            1.0 if status == Status.SLP else 0.0,
            1.0 if status == Status.FRZ else 0.0,
        ]
    except Exception:
        return [0.0] * 6


def _boost_vec(boosts: Optional[Dict[str, int]]) -> List[float]:
    """[atk, spa, spe, def, spd] normalized to [-1,1] (stage/6)."""
    if not boosts:
        return [0.0] * 5
    def _nb(k: str) -> float:
        return max(-1.0, min(1.0, float(boosts.get(k, 0)) / 6.0))
    return [_nb("atk"), _nb("spa"), _nb("spe"), _nb("def"), _nb("spd")]


def _move_flags(pokemon: Any) -> Tuple[float, float, float, float, float, float]:
    """(has_setup, has_recovery, has_priority, has_pivot, has_hazard_clear, move_count_known)."""
    moves = getattr(pokemon, "moves", None) or {}
    if not moves:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    has_setup = has_recovery = has_priority = has_pivot = has_hazard_clear = 0.0
    for mid, move_obj in moves.items():
        nid = _norm_id(mid)
        if nid in _SETUP_IDS:        has_setup = 1.0
        if nid in _RECOVERY_IDS:     has_recovery = 1.0
        if nid in _PIVOT_IDS:        has_pivot = 1.0
        if nid in _HAZARD_CLEAR_IDS: has_hazard_clear = 1.0
        if int(getattr(move_obj, "priority", 0) or 0) > 0:
            has_priority = 1.0
    return has_setup, has_recovery, has_priority, has_pivot, has_hazard_clear, min(1.0, len(moves) / 4.0)


def _best_dmg_frac(pokemon: Any, target: Any, battle: Any) -> float:
    """Best estimated damage / target max HP, capped at 1."""
    if _estimate_dmg is None or target is None or pokemon is None:
        return 0.0
    best = 0.0
    for move_obj in (getattr(pokemon, "moves", None) or {}).values():
        if float(getattr(move_obj, "base_power", 0) or 0) <= 0:
            continue
        try:
            d = min(1.0, max(0.0, float(_estimate_dmg(move_obj, pokemon, target, battle))))
            if d > best:
                best = d
        except Exception:
            pass
    return best


def _belief_entropy(belief: Any) -> float:
    """Normalized Shannon entropy [0,1] of an OpponentBelief distribution."""
    if belief is None:
        return 0.0
    dist = getattr(belief, "dist", None)
    if not dist:
        return 0.0
    ps = [max(0.0, float(p)) for _, p in dist]
    total = sum(ps)
    if total <= 1e-12:
        return 0.0
    ps = [p / total for p in ps]
    k = len(ps)
    if k <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in ps if p > 1e-12)
    return max(0.0, min(1.0, h / math.log(k)))


def _effective_speed(mon: Any, status: Any, boosts: Optional[Dict[str, int]]) -> float:
    """Simplified effective speed: stat + boost + paralysis + Choice Scarf."""
    if mon is None:
        return 0.0
    try:
        base = float((getattr(mon, "stats", None) or {}).get("spe", 100) or 100)
        stage = int((boosts or {}).get("spe", 0))
        if stage > 0:
            base *= (2 + stage) / 2.0
        elif stage < 0:
            base *= 2.0 / (2 + abs(stage))
        if _HAS_STATUS and status == Status.PAR:
            base *= 0.5
        item = str(getattr(mon, "item", "") or "").lower().replace(" ", "").replace("-", "")
        if "choicescarf" in item:
            base *= 1.5
        return base
    except Exception:
        return 100.0


def _volatile_features(
    pokemon: Any,
    is_active: bool,
    shadow_volatiles: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Extract 8 volatile-condition features for one Pokemon slot.

    Returns (has_substitute, has_leech_seed, is_taunted, is_encored,
             is_confused, perish_turns, yawn_incoming, is_terastallized).

    Sources (in priority order):
      1. pokemon.effects  — poke_env's real battle tracking (Effect enum keys)
      2. shadow_volatiles — ShadowState dict for the active pokemon (confusion)
      3. pokemon.is_terastallized — real battle or proxy
    """
    if not is_active or pokemon is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    has_sub = has_leech = is_taunt = is_encore = is_conf = 0.0
    perish = yawn = is_tera = 0.0

    # --- poke_env Effect dict ---
    if _HAS_EFFECT:
        try:
            effects = getattr(pokemon, "effects", {}) or {}
            if Effect.SUBSTITUTE  in effects: has_sub   = 1.0
            if Effect.LEECH_SEED  in effects: has_leech = 1.0
            if Effect.TAUNT       in effects: is_taunt  = 1.0
            if Effect.ENCORE      in effects: is_encore = 1.0
            if Effect.CONFUSION   in effects: is_conf   = 1.0
            if Effect.YAWN        in effects: yawn      = 1.0
            # Perish counter: PERISH3=3 turns left, PERISH2=2, PERISH1=1, PERISH0=next
            for turns_left, eff in [(3, Effect.PERISH3), (2, Effect.PERISH2),
                                    (1, Effect.PERISH1), (0, Effect.PERISH0)]:
                if eff in effects:
                    perish = float(turns_left) / 3.0
                    break
        except Exception:
            pass

    # --- ShadowState volatile dict (fallback / in-rollout) ---
    if shadow_volatiles is not None:
        try:
            if shadow_volatiles.get("confusion_turns", 0) > 0:
                is_conf = 1.0
        except Exception:
            pass

    # --- Tera ---
    try:
        if bool(getattr(pokemon, "is_terastallized", False)):
            is_tera = 1.0
    except Exception:
        pass

    return has_sub, has_leech, is_taunt, is_encore, is_conf, perish, yawn, is_tera


def _encode_one_mon(
    pokemon: Any,
    hp_frac: float,
    status: Any,
    boosts: Optional[Dict[str, int]],
    is_active: bool,
    active_opp: Any,
    battle: Any,
    belief: Any = None,
    shadow_volatiles: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Encode a single Pokemon slot into N_MON_FEATURES floats."""
    vec = np.zeros(N_MON_FEATURES, dtype=np.float32)

    hp = max(0.0, min(1.0, float(hp_frac)))
    vec[0] = hp
    vec[1] = 1.0 if is_active else 0.0
    vec[2] = 1.0 if hp <= 0.0 else 0.0

    vec[3:9] = _status_vec(status)

    bv = _boost_vec(boosts)
    vec[9], vec[10], vec[11], vec[12], vec[13] = bv[0], bv[1], bv[2], bv[3], bv[4]

    if active_opp is not None:
        vec[14] = _max_type_adv(pokemon, active_opp)
        vec[15] = _max_type_adv(active_opp, pokemon)
    else:
        vec[14] = vec[15] = 0.25

    if hp > 0.0:
        vec[16] = _best_dmg_frac(pokemon, active_opp, battle)

    flags = _move_flags(pokemon)
    vec[17], vec[18], vec[19], vec[20], vec[21], vec[22] = flags

    vec[23] = _belief_entropy(belief)

    # Volatile conditions (features 24-31)
    vol = _volatile_features(pokemon, is_active, shadow_volatiles)
    vec[24], vec[25], vec[26], vec[27], vec[28], vec[29], vec[30], vec[31] = vol

    return vec


def _unseen_slot_vec() -> np.ndarray:
    """Feature vector for a completely unseen opponent slot."""
    vec = np.zeros(N_MON_FEATURES, dtype=np.float32)
    vec[0]  = 1.0   # assume full health
    vec[14] = 0.25  # neutral type matchup
    vec[15] = 0.25
    vec[23] = 1.0   # maximum uncertainty
    return vec


def _sorted_team(team: List[Any], active: Any, hp_map: Dict[int, float]) -> List[Any]:
    """Active first, bench alive, fainted last, padded to 6 with None."""
    alive_bench = [p for p in team if p is not None and p is not active and hp_map.get(id(p), 1.0) > 0.0]
    fainted     = [p for p in team if p is not None and p is not active and hp_map.get(id(p), 1.0) <= 0.0]
    ordered = ([active] if active is not None else []) + alive_bench + fainted
    while len(ordered) < 6:
        ordered.append(None)
    return ordered[:6]


def _encode_field(
    state: Any,
    my_team: List[Any],
    my_active: Any,
    opp_active: Any,
    my_hp: Dict,
    opp_hp: Dict,
    my_status: Dict,
    opp_status: Dict,
    my_boosts: Dict,
    opp_boosts: Dict,
    battle: Any,
) -> np.ndarray:
    """Encode global battlefield features (N_FIELD_FEATURES,)."""
    vec = np.zeros(N_FIELD_FEATURES, dtype=np.float32)

    # --- Weather (0-3) ---
    for k, v in ((getattr(state, "shadow_weather", None) or {}).items()):
        if not (isinstance(v, (int, float)) and v > 0):
            continue
        name = getattr(k, "name", str(k)).lower().replace("_", "")
        if   "sunnyday"    in name or "desolateland"  in name: vec[0] = 1.0
        elif "raindance"   in name or "primordialsea" in name: vec[1] = 1.0
        elif "sandstorm"   in name:                            vec[2] = 1.0
        elif "hail"        in name or "snow"          in name: vec[3] = 1.0

    # --- Terrain (4-7) ---
    shadow_fields = getattr(state, "shadow_fields", None) or {}
    for k, v in shadow_fields.items():
        if not (isinstance(v, (int, float)) and v > 0):
            continue
        name = getattr(k, "name", str(k)).lower().replace("_", "")
        if   "electricterrain" in name: vec[4] = 1.0
        elif "grassyterrain"   in name: vec[5] = 1.0
        elif "psychicterrain"  in name: vec[6] = 1.0
        elif "mistyterrain"    in name: vec[7] = 1.0

    # --- Side conditions (8-15) ---
    def _apply_sc(sc_dict: Any, offset: int) -> None:
        for k, v in ((sc_dict or {}).items() if hasattr(sc_dict, "items") else []):
            name = getattr(k, "name", str(k)).lower().replace("_", "").replace(" ", "")
            val  = float(v) if isinstance(v, (int, float)) else 1.0
            if   "stealthrock" in name:                                      vec[offset+0] = 1.0
            elif "spikes" in name and "toxic" not in name and "sticky" not in name:
                vec[offset+1] = min(1.0, val / 3.0)
            elif "toxicspikes" in name:                                      vec[offset+2] = min(1.0, val / 2.0)
            elif "stickyweb"   in name:                                      vec[offset+3] = 1.0

    _apply_sc(getattr(state, "my_side_conditions",  None),  8)
    _apply_sc(getattr(state, "opp_side_conditions", None), 12)

    # --- Team-level summary signals (16-20) ---

    # 16: moves_first
    trick_room = any(
        "trickroom" in getattr(k, "name", str(k)).lower().replace("_", "")
        for k, v in shadow_fields.items()
        if isinstance(v, (int, float)) and v > 0
    )
    my_spe  = _effective_speed(my_active,  my_status.get(id(my_active))  if my_active  else None, my_boosts.get(id(my_active))  if my_active  else None)
    opp_spe = _effective_speed(opp_active, opp_status.get(id(opp_active)) if opp_active else None, opp_boosts.get(id(opp_active)) if opp_active else None)
    if trick_room:
        my_spe, opp_spe = opp_spe, my_spe
    vec[16] = 1.0 if my_spe > opp_spe else (-1.0 if opp_spe > my_spe else 0.0)

    # 17: team_hp_diff = (sum my alive hp - sum opp alive hp) / 6
    my_hp_sum  = sum(max(0.0, min(1.0, v)) for v in my_hp.values())
    opp_hp_sum = sum(max(0.0, min(1.0, v)) for v in opp_hp.values())
    vec[17] = max(-1.0, min(1.0, (my_hp_sum - opp_hp_sum) / 6.0))

    # 18: alive_count_diff = (my_alive - opp_alive) / 6
    my_alive   = sum(1 for v in my_hp.values()  if v > 0)
    opp_alive  = sum(1 for v in opp_hp.values() if v > 0)
    opp_unseen = sum(1 for s in (getattr(state, "opp_slots", []) or []) if s is None)
    vec[18] = max(-1.0, min(1.0, (my_alive - (opp_alive + opp_unseen)) / 6.0))

    # 19: best_switch_safety = max over alive bench of (1 - expected_damage_taken_frac)
    best_safety = 0.0
    for mon in my_team:
        if mon is None or mon is my_active or my_hp.get(id(mon), 0.0) <= 0.0:
            continue
        dmg_taken = _best_dmg_frac(opp_active, mon, battle) if opp_active is not None else 0.0
        safety = max(0.0, min(1.0, 1.0 - dmg_taken))
        if safety > best_safety:
            best_safety = safety
    vec[19] = best_safety

    # 20: hazard_pressure — weighted net hazard load in your favor.
    my_load  = vec[8]  * 0.40 + vec[9]  * 0.40 + vec[10] * 0.15 + vec[11] * 0.05
    opp_load = vec[12] * 0.40 + vec[13] * 0.40 + vec[14] * 0.15 + vec[15] * 0.05
    vec[20] = max(-1.0, min(1.0, opp_load - my_load))

    # --- Tera usage (21-22) ---
    # Read from real battle object when available; ShadowState doesn't track this.
    if battle is not None:
        try:
            vec[21] = 1.0 if bool(getattr(battle, "used_tera", False)) else 0.0
        except Exception:
            pass
        try:
            vec[22] = 1.0 if bool(getattr(battle, "opponent_used_tera", False)) else 0.0
        except Exception:
            pass

    return vec


def _encode_active_pair(
    my_active: Any,
    opp_active: Any,
    my_hp: Dict,
    opp_hp: Dict,
    my_spe: float,
    opp_spe: float,
    battle: Any,
) -> np.ndarray:
    """
    Encode the active-vs-active tactical state into N_ACTIVE_PAIR_FEATURES floats.
    """
    vec = np.zeros(N_ACTIVE_PAIR_FEATURES, dtype=np.float32)

    my_hp_frac  = my_hp.get(id(my_active),   0.0) if my_active  is not None else 0.0
    opp_hp_frac = opp_hp.get(id(opp_active), 0.0) if opp_active is not None else 0.0

    if my_active is not None and opp_active is not None:
        my_dmg  = _best_dmg_frac(my_active,  opp_active, battle)
        vec[0]  = 1.0 if my_hp_frac > 0.0 and my_dmg  >= opp_hp_frac > 0.0 else 0.0

        opp_dmg = _best_dmg_frac(opp_active, my_active,  battle)
        vec[1]  = 1.0 if opp_hp_frac > 0.0 and opp_dmg >= my_hp_frac  > 0.0 else 0.0

    vec[2] = 1.0 if my_spe == opp_spe else 0.0

    return vec


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode_state(state: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode a ShadowState into the three feature arrays.

    Returns:
        tokens     : float32 (12, N_MON_FEATURES)
        field      : float32 (N_FIELD_FEATURES,)
        active_pair: float32 (N_ACTIVE_PAIR_FEATURES,)
    """
    tokens = np.zeros((12, N_MON_FEATURES), dtype=np.float32)

    battle      = getattr(state, "battle",       None)
    my_active   = getattr(state, "my_active",    None)
    opp_active  = getattr(state, "opp_active",   None)
    my_hp       = getattr(state, "my_hp",        {}) or {}
    opp_hp      = getattr(state, "opp_hp",       {}) or {}
    my_status   = getattr(state, "my_status",    {}) or {}
    opp_status  = getattr(state, "opp_status",   {}) or {}
    my_boosts   = getattr(state, "my_boosts",    {}) or {}
    opp_boosts  = getattr(state, "opp_boosts",   {}) or {}
    opp_beliefs = getattr(state, "opp_beliefs",  {}) or {}
    my_volatiles  = getattr(state, "my_volatiles",  {}) or {}
    opp_volatiles = getattr(state, "opp_volatiles", {}) or {}

    # --- My team (slots 0-5) ---
    my_team = list(getattr(state, "my_team", []) or [])
    for i, mon in enumerate(_sorted_team(my_team, my_active, my_hp)):
        if mon is None:
            continue
        tokens[i] = _encode_one_mon(
            mon, my_hp.get(id(mon), 1.0), my_status.get(id(mon)),
            my_boosts.get(id(mon)), is_active=(mon is my_active),
            active_opp=opp_active, battle=battle,
            shadow_volatiles=my_volatiles.get(id(mon)) if mon is my_active else None,
        )

    # --- Opponent team (slots 6-11): active always at index 6 ---
    opp_slots  = list(getattr(state, "opp_slots", []) or [])
    active_idx = int(getattr(state, "opp_active_idx", 0) or 0)

    if opp_slots:
        active_mon = opp_slots[active_idx] if active_idx < len(opp_slots) else None
        rest = [opp_slots[k] for k in range(len(opp_slots)) if k != active_idx]
        opp_ordered = ([active_mon] + rest + [None] * 6)[:6]
        for i, mon in enumerate(opp_ordered):
            if mon is None:
                tokens[6 + i] = _unseen_slot_vec()
            else:
                tokens[6 + i] = _encode_one_mon(
                    mon, opp_hp.get(id(mon), 1.0), opp_status.get(id(mon)),
                    opp_boosts.get(id(mon)), is_active=(mon is opp_active),
                    active_opp=my_active, battle=battle,
                    belief=opp_beliefs.get(id(mon)),
                    shadow_volatiles=opp_volatiles.get(id(mon)) if mon is opp_active else None,
                )
    else:
        opp_team = list(getattr(state, "opp_team", []) or [])
        for i, mon in enumerate(_sorted_team(opp_team, opp_active, opp_hp)):
            if mon is None:
                tokens[6 + i] = _unseen_slot_vec()
            else:
                tokens[6 + i] = _encode_one_mon(
                    mon, opp_hp.get(id(mon), 1.0), opp_status.get(id(mon)),
                    opp_boosts.get(id(mon)), is_active=(mon is opp_active),
                    active_opp=my_active, battle=battle,
                    belief=opp_beliefs.get(id(mon)),
                    shadow_volatiles=opp_volatiles.get(id(mon)) if mon is opp_active else None,
                )

    field = _encode_field(
        state, my_team, my_active, opp_active,
        my_hp, opp_hp, my_status, opp_status, my_boosts, opp_boosts, battle,
    )

    my_spe  = _effective_speed(my_active,  my_status.get(id(my_active))  if my_active  else None, my_boosts.get(id(my_active))  if my_active  else None)
    opp_spe = _effective_speed(opp_active, opp_status.get(id(opp_active)) if opp_active else None, opp_boosts.get(id(opp_active)) if opp_active else None)

    active_pair = _encode_active_pair(
        my_active, opp_active, my_hp, opp_hp, my_spe, opp_spe, battle,
    )

    return tokens, field, active_pair


def encode_state_flat(state: Any) -> np.ndarray:
    """
    Encode a ShadowState into a flat (N_TOTAL,) = (410,) float32 vector.

    Layout: [token_0, ..., token_11, field_features, active_pair_features]
    """
    tokens, field, active_pair = encode_state(state)
    return np.concatenate([tokens.reshape(-1), field, active_pair]).astype(np.float32)
