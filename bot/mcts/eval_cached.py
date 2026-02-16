import math
from typing import Any, Optional
from poke_env.battle import Status

# These will be imported from your actual eval.py
# For now we'll define the interface
from mcts_cache import MCTSCache, create_cached_role_weight_fn, create_cached_damage_fn


def _tanh01(x: float) -> float:
    return math.tanh(float(x))


def _team_hp_sum(hp_map: dict) -> float:
    return float(sum(max(0.0, min(1.0, v)) for v in hp_map.values()))


def low_hp_multiplier(hp: float, role_w: float) -> float:
    """Cached-friendly version (no external dependencies)."""
    if hp < 0.20:
        return 0.35 if role_w > 1.06 else 0.45
    if hp < 0.35:
        return 0.55 if role_w > 1.06 else 0.65
    if hp < 0.55:
        return 0.80
    return 1.00


def boost_multiplier(boosts: dict, hp: float) -> float:
    """Cached-friendly version."""
    if not boosts:
        return 1.00
    max_pos = max((v for v in boosts.values() if v > 0), default=0)
    if hp < 0.35:
        max_pos = min(max_pos, 2)
    if max_pos >= 4:
        return 1.18
    if max_pos >= 2:
        return 1.12
    if max_pos >= 1:
        return 1.06
    return 1.00


def status_multiplier(status) -> float:
    """Cached-friendly version."""
    if status in (Status.TOX, Status.PSN):
        return 0.85
    if status == Status.BRN:
        return 0.88
    if status == Status.PAR:
        return 0.93
    return 1.00


def team_value(
    team,
    hp_map,
    boosts_map,
    gen: int,
    status_map=None,
    cache: Optional[MCTSCache] = None,
    role_weight_fn=None,
) -> float:
    """
    Calculate team value with caching support.
    
    Args:
        team: List of Pokemon
        hp_map: Dict of pokemon_id -> hp_fraction
        boosts_map: Dict of pokemon_id -> boost_dict
        gen: Generation number
        status_map: Optional dict of pokemon_id -> Status
        cache: Optional MCTSCache for role weight lookups
        role_weight_fn: Function to compute role weights
    """
    total = 0.0
    status_map = status_map or {}
    
    # Alive list for "only X left" checks
    alive = [m for m in team if float(hp_map.get(id(m), 0.0)) > 0.0]
    
    # Use cached property lookups if available
    if cache:
        num_removers = sum(1 for m in alive if cache.get_mon_properties(m).get('has_removal', False))
        num_priority = sum(1 for m in alive if cache.get_mon_properties(m).get('has_priority', False))
    else:
        # Fallback to original logic
        from bot.mcts.eval import _mon_has_removal, _mon_has_damaging_priority
        num_removers = sum(1 for m in alive if _mon_has_removal(m))
        num_priority = sum(1 for m in alive if _mon_has_damaging_priority(m))
    
    for mon in team:
        hp = float(hp_map.get(id(mon), 0.0))
        if hp <= 0.0:
            continue
        
        # Get role weight (cached if cache provided)
        if cache and role_weight_fn:
            role_w = cache.get_role_weight(mon, gen, lambda m, g: role_weight_fn(m, g))
        elif role_weight_fn:
            role_w = role_weight_fn(mon, gen)
        else:
            role_w = 1.0
        
        boosts = (boosts_map.get(id(mon), {}) if boosts_map else {}) or {}
        st = status_map.get(id(mon), None)
        
        # Unique-role multiplier
        unique_mult = 1.0
        try:
            if cache:
                has_removal = cache.get_mon_properties(mon).get('has_removal', False)
                has_priority = cache.get_mon_properties(mon).get('has_priority', False)
            else:
                from bot.mcts.eval import _mon_has_removal, _mon_has_damaging_priority
                has_removal = _mon_has_removal(mon)
                has_priority = _mon_has_damaging_priority(mon)
            
            if num_removers == 1 and has_removal:
                unique_mult *= 1.10
            if num_priority == 1 and has_priority:
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


def evaluate_boosts(state: Any) -> float:
    """Boost evaluation (no caching needed - already fast)."""
    my_active_id = id(state.my_active)
    opp_active_id = id(state.opp_active)
    
    my_boosts = state.my_boosts.get(my_active_id, {})
    opp_boosts = state.opp_boosts.get(opp_active_id, {})
    
    my_value = _calculate_boost_state_value(my_boosts)
    opp_value = _calculate_boost_state_value(opp_boosts)
    
    diff = my_value - opp_value
    return _tanh01(diff / 10.0)


def _calculate_boost_state_value(boosts: dict) -> float:
    """Calculate boost value with diminishing returns."""
    value = 0.0
    
    for stat in ['atk', 'spa', 'spe', 'def', 'spd']:
        stages = boosts.get(stat, 0)
        
        if stat in ['atk', 'spa']:
            base_weight = 1.5
        elif stat == 'spe':
            base_weight = 1.2
        else:
            base_weight = 0.7
        
        if stages > 0:
            stage_value = 0.0
            for i in range(1, stages + 1):
                if i == 1:
                    stage_value += 1.0
                elif i == 2:
                    stage_value += 0.8
                elif i == 3:
                    stage_value += 0.6
                elif i == 4:
                    stage_value += 0.4
                elif i == 5:
                    stage_value += 0.2
                else:
                    stage_value += 0.1
            
            value += stage_value * base_weight
        
        elif stages < 0:
            value += stages * base_weight
    
    return value


def evaluate_state_cached(
    state: Any,
    cache: Optional[MCTSCache] = None,
    role_weight_fn=None,
) -> float:
    """
    Cached version of evaluate_state().
    
    Args:
        state: ShadowState to evaluate
        cache: Optional MCTSCache for lookups
        role_weight_fn: Function to compute role weights (will be cached)
    
    Returns:
        Evaluation score in [-1, +1]
    """
    
    # Hard terminal checks
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
    
    # Safety check
    if my_active_hp <= 0.0 and opp_active_hp > 0.0:
        lead_hint = _tanh01((my_sum_raw - opp_sum_raw) / 1.5)
        return max(-1.0, min(1.0, float(-0.90 + 0.15 * lead_hint)))
    
    gen = int(getattr(state.battle, "gen", 9) or 9)
    
    with state._patched_status(), state._patched_boosts():
        # Team value with caching
        my_value = team_value(
            state.my_team, state.my_hp, state.my_boosts, gen,
            status_map=state.my_status,
            cache=cache,
            role_weight_fn=role_weight_fn,
        )
        
        opp_value_known = team_value(
            state.opp_team, state.opp_hp, state.opp_boosts, gen,
            status_map=state.opp_status,
            cache=cache,
            role_weight_fn=role_weight_fn,
        )
        
        # Import unseen value calculation
        from bot.mcts.eval import opp_unseen_value
        opp_unseen = max(0, opp_total - opp_known)
        opp_value = opp_value_known + opp_unseen_value(opp_known, opp_total)
        
        team_term = _tanh01((my_value - opp_value) / 1.2)
        
        # Numbers advantage
        from bot.mcts.eval import healthy_count
        my_healthy = healthy_count(state.my_team, state.my_hp, 0.55)
        opp_healthy = healthy_count(state.opp_team, state.opp_hp, 0.55) + opp_unseen
        numbers_term = _tanh01((my_healthy - opp_healthy) / 1.5)
        
        # Race advantage (with caching for damage if available)
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
            from bot.scoring.race import evaluate_race_for_move
            race = evaluate_race_for_move(state.battle, state.ctx_me, best_mv)
            race_term = _tanh01((race.ttd_me - race.tko_opp) / 1.5)
        
        # Switch term
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
            
            SW_NORM = 35.0
            switch_term = _tanh01(best_sw_score / SW_NORM)
        
        # Boost term
        boost_term = evaluate_boosts(state)
        if my_active_hp < 0.20:
            boost_term *= 0.40
        elif my_active_hp < 0.35:
            boost_term *= 0.70
        
        # Status term
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
        
        # Active preservation (with cached role weight)
        if cache and role_weight_fn:
            my_active_role = cache.get_role_weight(state.my_active, gen, lambda m, g: role_weight_fn(m, g))
        elif role_weight_fn:
            my_active_role = role_weight_fn(state.my_active, gen)
        else:
            my_active_role = 1.0
        
        if my_active_role > 1.06:
            active_preserve = _tanh01((my_active_hp - 0.60) / 0.20)
        else:
            active_preserve = _tanh01((my_active_hp - 0.45) / 0.25)
    
    # Tempo penalty
    tempo_penalty = 0.04 * float(state.ply)
    
    # Progress term
    opp_sum_now = _team_hp_sum(state.opp_hp)
    progress_term = _tanh01((1.0 - opp_sum_now) / 0.6)
    
    # Material lead
    my_alive = sum(1 for v in state.my_hp.values() if float(v) > 0.0)
    opp_alive = sum(1 for v in state.opp_hp.values() if float(v) > 0.0)
    ahead = my_alive - opp_alive
    
    ahead_factor = max(0.0, min(1.0, (ahead - 1) / 3.0))
    active_preserve *= (1.0 - 0.50 * ahead_factor)
    
    # Sac penalty
    sac_penalty = 0.0
    if ahead >= 2:
        if my_active_hp <= 0.0:
            if opp_active_hp <= 0.0:
                sac_penalty += 0.02
            else:
                sac_penalty += 0.20
        elif my_active_hp < 0.15:
            sac_penalty += 0.10
        elif my_active_hp < 0.30:
            sac_penalty += 0.05
    
    # Dynamic weights
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