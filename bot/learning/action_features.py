"""
Per-action feature computation shared by data collection and inference.

Feature vector (26 values, fixed order = ACTION_FEATURE_NAMES):

  # | Name                    | Description
  --+-------------------------+-----------------------------------------------------
  0 | bp_norm                 | base_power / 150, capped at 1.  0 for switches.
  1 | accuracy                | float 0-1. True (always-hit) -> 1.0.  0 for sw.
  2 | priority_pos            | 1 if priority > 0, else 0.  0 for sw.
  3 | priority_neg            | 1 if priority < 0, else 0.  0 for sw.
  4 | is_physical             | 1 if MoveCategory.PHYSICAL.  0 for sw.
  5 | is_special              | 1 if MoveCategory.SPECIAL.  0 for sw.
  6 | is_status               | 1 if MoveCategory.STATUS.  0 for sw.
  7 | stab                    | 1 if move type in active pokemon's types.  0 for sw.
  8 | dmg_frac                | estimate_damage_fraction capped at 1.  0 for sw.
  9 | ko_flag                 | 1 if dmg_frac >= opp current_hp_fraction.  0 for sw.
 10 | is_recovery             | 1 if recovery move (recover/roost/synthesis/...). 0 for sw.
 11 | is_hazard_clear         | 1 if defog or rapid spin.  0 for sw.
 12 | is_setup                | 1 if setup move (SD/NP/DD/CM/...).  0 for sw.
 13 | is_disruption           | 1 if anti-tempo move (taunt/haze/roar/encore/...). 0 for sw.
 14 | sw_hp_frac              | switch-in's current_hp_fraction.  0 for moves.
 15 | sw_hazard_dmg           | compute_switch_tax(side_conditions).  0 for moves.
 16 | sw_type_disadv          | max opp type-eff vs switch-in / 4.0.  0 for moves.
 17 | sw_type_adv             | max switch-in type-eff vs opp / 4.0.  0 for moves.
 18 | sw_is_pivot             | 1 if switch-in has a pivot move.  0 for moves.
 19 | sw_can_remove_hazards   | 1 if switch-in has defog or rapid spin.  0 for moves.
 20 | sw_resists_opp_moves    | fraction of revealed opp moves switch-in resists (<1x). 0 for moves.
 21 | sw_expected_dmg_taken   | top-2 avg dmg fraction from revealed opp damaging moves. 0 for moves.
 22 | opp_revealed_move_count | # revealed opp moves / 4.  Non-zero for both moves and switches.
 23 | sw_known_move_count     | # confirmed moves on switch-in / 4.  0 for moves.
 24 | is_switch               | 1 if switch, 0 if move.
 25 | heuristic_score         | raw heuristic logit (prior_raw from score_move/score_switch).
                               | NOTE: compute_action_features() always returns 0.0 for this;
                               | callers must set it from prior_raw after calling this function.

Notes:
- Multiple flags can be 1 for the same move (e.g. mortal spin = hazard_clear + status).
- opp_revealed_move_count disambiguates "0 = resists nothing" from "0 = no info".
- sw_known_move_count is always reliable for OUR pokemon (full moveset known); it
  disambiguates "0 = no pivot" from "0 = moveset not seen yet" (relevant in formats
  where bench moves are hidden, though less so in random battles).
"""
from __future__ import annotations

from typing import Any, Dict, List

try:
    from poke_env.battle import MoveCategory
    from poke_env.data import GenData
    _TYPE_CHART = GenData.from_gen(9).type_chart
except Exception:
    MoveCategory = None  # type: ignore
    _TYPE_CHART = {}

try:
    from bot.scoring.damage_score import estimate_damage_fraction
except Exception:
    estimate_damage_fraction = None  # type: ignore

try:
    from bot.mcts.eval import compute_switch_tax
except Exception:
    compute_switch_tax = None  # type: ignore


# ---------------------------------------------------------------------------
# Move category sets (normalised: lowercase, no spaces/hyphens/underscores)
# ---------------------------------------------------------------------------

_RECOVERY_MOVE_IDS = frozenset({
    "recover", "softboiled", "slackoff", "roost", "moonlight", "morningsun",
    "synthesis", "shoreup", "leechseed", "strengthsap", "oblivionwing",
    "healorder", "milkdrink", "lifedew", "lunarblessing",
})

_HAZARD_CLEAR_IDS = frozenset({"rapidspin", "defog"})

_SETUP_MOVE_IDS = frozenset({
    "swordsdance", "nastyplot", "dragondance", "calmmind", "bulkup",
    "quiverdance", "shellsmash", "bellydrum", "shiftgear", "agility",
    "tailglow", "coil", "curse", "growth", "workup", "honeclaws",
    "cottonguard", "irondefense", "acidarmor",
})

# Anti-setup / anti-tempo control only.
# Excludes partingshot (pivot) and memento (self-sacrifice) — different decision type.
_DISRUPTION_IDS = frozenset({
    "haze", "clearsmog", "roar", "whirlwind", "dragontail",
    "encore", "taunt", "torment", "disable",
})

_PIVOT_MOVE_IDS = frozenset({
    "uturn", "voltswitch", "flipturn", "partingshot", "chillyreception", "teleport",
})


def _norm_move_id(mid: Any) -> str:
    s = getattr(mid, "id", None) or getattr(mid, "name", None) or str(mid)
    return str(s).lower().replace("_", "").replace("-", "").replace(" ", "")


ACTION_FEATURE_NAMES: List[str] = [
    "bp_norm",
    "accuracy",
    "priority_pos",
    "priority_neg",
    "is_physical",
    "is_special",
    "is_status",
    "stab",
    "dmg_frac",
    "ko_flag",
    "is_recovery",
    "is_hazard_clear",
    "is_setup",
    "is_disruption",
    "sw_hp_frac",
    "sw_hazard_dmg",
    "sw_type_disadv",
    "sw_type_adv",
    "sw_is_pivot",
    "sw_can_remove_hazards",
    "sw_resists_opp_moves",
    "sw_expected_dmg_taken",
    "opp_revealed_move_count",
    "sw_known_move_count",
    "is_switch",
    "heuristic_score",  # set externally from prior_raw; always 0.0 from compute_action_features()
]

_N_FEATURES = len(ACTION_FEATURE_NAMES)  # 26


def _safe_hp_frac(mon: Any) -> float:
    try:
        frac = getattr(mon, "current_hp_fraction", None)
        if frac is not None:
            return float(frac)
        hp = float(getattr(mon, "current_hp", 0) or 0)
        max_hp = float(getattr(mon, "max_hp", 1) or 1)
        return hp / max_hp if max_hp > 0 else 0.0
    except Exception:
        return 1.0


def _type_effectiveness(move_type: Any, defender: Any) -> float:
    """Damage multiplier for move_type vs defender's types. Falls back to 1.0."""
    try:
        def_types = [t for t in (getattr(defender, "types", None) or []) if t is not None]
        if not def_types or not _TYPE_CHART:
            return 1.0
        mult = 1.0
        atk_name = getattr(move_type, "name", str(move_type)).lower()
        for dt in def_types:
            dt_name = getattr(dt, "name", str(dt)).lower()
            row = _TYPE_CHART.get(atk_name, {})
            mult *= float(row.get(dt_name, 1.0))
        return mult
    except Exception:
        return 1.0


def _max_type_effectiveness_vs(attacker: Any, defender: Any) -> float:
    """Maximum single-type effectiveness of any of attacker's types vs defender."""
    try:
        atk_types = [t for t in (getattr(attacker, "types", None) or []) if t is not None]
        if not atk_types:
            return 1.0
        best = 0.0
        for t in atk_types:
            eff = _type_effectiveness(t, defender)
            if eff > best:
                best = eff
        return best
    except Exception:
        return 1.0


def _has_move_in_set(mon: Any, move_set: frozenset) -> bool:
    """True if mon has any confirmed move whose normalised id is in move_set."""
    moves = getattr(mon, "moves", None) or {}
    for mid in moves.keys():
        if _norm_move_id(mid) in move_set:
            return True
    return False


def compute_action_features(
    kind: str,
    obj: Any,
    battle: Any,
    ctx_me: Any,
) -> Dict[str, float]:
    """
    Returns a dict mapping ACTION_FEATURE_NAMES -> float for one action.

    kind:   "move" or "switch"
    obj:    Move object (for moves) or Pokemon object (for switches)
    battle: poke-env Battle (used for dmg calc and side_conditions)
    ctx_me: EvalContext with .me = our active Pokemon, .opp = opp active

    Returns all zeros on any error, so callers never need to guard.
    """
    out: Dict[str, float] = {k: 0.0 for k in ACTION_FEATURE_NAMES}
    try:
        opp = getattr(ctx_me, "opp", None)

        # Context feature: how many of the opponent's moves are revealed (0–4 / 4).
        # Non-zero for both move and switch actions — disambiguates "unknown" from "0".
        opp_moves = list((getattr(opp, "moves", {}) or {}).values()) if opp is not None else []
        out["opp_revealed_move_count"] = min(1.0, len(opp_moves) / 4.0)

        if kind == "switch":
            # --- Switch features ---
            out["is_switch"] = 1.0

            sw_in = obj  # the Pokemon we're switching in

            out["sw_hp_frac"] = _safe_hp_frac(sw_in)

            # Confirmed moves on the switch-in (always known for our own pokemon).
            sw_moves = list((getattr(sw_in, "moves", {}) or {}).values())
            out["sw_known_move_count"] = min(1.0, len(sw_moves) / 4.0)

            # Hazard entry damage for our side
            if compute_switch_tax is not None:
                side_conds = getattr(battle, "side_conditions", {}) or {}
                norm_sc: Dict[str, int] = {}
                for k, v in side_conds.items():
                    key = getattr(k, "name", str(k)).lower().replace("_", "").replace(" ", "")
                    norm_sc[key] = int(v) if isinstance(v, int) else 1
                try:
                    out["sw_hazard_dmg"] = float(compute_switch_tax(norm_sc))
                except Exception:
                    pass

            if opp is not None:
                # Type (dis)advantage
                disadv = _max_type_effectiveness_vs(opp, sw_in)
                out["sw_type_disadv"] = min(1.0, disadv / 4.0)

                adv = _max_type_effectiveness_vs(sw_in, opp)
                out["sw_type_adv"] = min(1.0, adv / 4.0)

                # Resistance fraction: pure type-chart signal, no BP weighting.
                # 0 both when "resists nothing" and "no moves revealed" — the mask
                # feature opp_revealed_move_count tells them apart.
                if opp_moves:
                    resists = sum(
                        1 for m in opp_moves
                        if _type_effectiveness(getattr(m, "type", None), sw_in) < 1.0
                    )
                    out["sw_resists_opp_moves"] = resists / len(opp_moves)

                # Expected damage taken: top-2 average over revealed DAMAGING moves.
                # Using top-2 avg instead of max to reduce noise from one spiky calc.
                if opp_moves and estimate_damage_fraction is not None:
                    damages = []
                    for opp_move in opp_moves:
                        bp = float(getattr(opp_move, "base_power", 0) or 0)
                        if bp <= 0:
                            continue  # status moves deal no direct damage
                        try:
                            d = float(estimate_damage_fraction(opp_move, opp, sw_in, battle))
                            damages.append(min(1.0, max(0.0, d)))
                        except Exception:
                            pass
                    if damages:
                        damages.sort(reverse=True)
                        top2 = damages[:2]
                        out["sw_expected_dmg_taken"] = sum(top2) / len(top2)

            # Structural utility flags (based on confirmed moveset)
            out["sw_is_pivot"] = 1.0 if _has_move_in_set(sw_in, _PIVOT_MOVE_IDS) else 0.0
            out["sw_can_remove_hazards"] = 1.0 if _has_move_in_set(sw_in, _HAZARD_CLEAR_IDS) else 0.0

        else:
            # --- Move features ---
            move = obj
            mid = _norm_move_id(move)

            bp = float(getattr(move, "base_power", 0) or 0)
            out["bp_norm"] = min(1.0, bp / 150.0)

            acc = getattr(move, "accuracy", 1.0)
            if acc is True or acc == 1:
                out["accuracy"] = 1.0
            elif isinstance(acc, (int, float)) and acc > 1:
                out["accuracy"] = float(acc) / 100.0
            else:
                out["accuracy"] = float(acc) if acc else 1.0

            priority = int(getattr(move, "priority", 0) or 0)
            out["priority_pos"] = 1.0 if priority > 0 else 0.0
            out["priority_neg"] = 1.0 if priority < 0 else 0.0

            if MoveCategory is not None:
                cat = getattr(move, "category", None)
                if cat == MoveCategory.PHYSICAL:
                    out["is_physical"] = 1.0
                elif cat == MoveCategory.SPECIAL:
                    out["is_special"] = 1.0
                elif cat == MoveCategory.STATUS:
                    out["is_status"] = 1.0

            # STAB
            me = getattr(ctx_me, "me", None)
            if me is not None:
                move_type = getattr(move, "type", None)
                if move_type is not None:
                    my_types = set(getattr(me, "types", None) or [])
                    if move_type in my_types:
                        out["stab"] = 1.0

            # Damage estimate
            if me is not None and opp is not None and estimate_damage_fraction is not None:
                try:
                    dmg = float(estimate_damage_fraction(move, me, opp, battle))
                    out["dmg_frac"] = min(1.0, max(0.0, dmg))
                    opp_hp = _safe_hp_frac(opp)
                    out["ko_flag"] = 1.0 if out["dmg_frac"] >= opp_hp else 0.0
                except Exception:
                    pass

            # Move role categories — multiple flags allowed (e.g. mortal spin = hazard_clear + status)
            out["is_recovery"]     = 1.0 if mid in _RECOVERY_MOVE_IDS else 0.0
            out["is_hazard_clear"] = 1.0 if mid in _HAZARD_CLEAR_IDS else 0.0
            out["is_setup"]        = 1.0 if mid in _SETUP_MOVE_IDS else 0.0
            out["is_disruption"]   = 1.0 if mid in _DISRUPTION_IDS else 0.0

    except Exception:
        # Return all zeros rather than propagating errors into the training loop
        return {k: 0.0 for k in ACTION_FEATURE_NAMES}

    return out
