import math
from typing import Any

from poke_env.battle import Status
from bot.scoring.race import evaluate_race_for_move


def _tanh01(x: float) -> float:
    # maps to (-1, 1)
    return math.tanh(float(x))


def _team_hp_sum(hp_map: dict[int, float]) -> float:
    return float(sum(max(0.0, min(1.0, v)) for v in hp_map.values()))

def evaluate_boosts(state: Any) -> float:
    """
    Evaluate stat boost advantage.
    
    Returns value in [-1, +1]:
    - Positive: We have boost advantage
    - Negative: Opponent has boost advantage

    NOTE: This is a fairly rudimentary approach, there's much better way to decide if the boosts on our side are better, will improve this later
    """
    my_active_id = id(state.my_active)
    opp_active_id = id(state.opp_active)
    
    my_boosts = state.my_boosts.get(my_active_id, {})
    opp_boosts = state.opp_boosts.get(opp_active_id, {})
    
    # Weight boosts by importance
    my_value = (
        my_boosts.get('atk', 0) * 1.5 +
        my_boosts.get('spa', 0) * 1.5 +
        my_boosts.get('spe', 0) * 1.2 +
        my_boosts.get('def', 0) * 0.7 +
        my_boosts.get('spd', 0) * 0.7
    )
    
    opp_value = (
        opp_boosts.get('atk', 0) * 1.5 +
        opp_boosts.get('spa', 0) * 1.5 +
        opp_boosts.get('spe', 0) * 1.2 +
        opp_boosts.get('def', 0) * 0.7 +
        opp_boosts.get('spd', 0) * 0.7
    )
    
    diff = my_value - opp_value
    return _tanh01(diff / 10.0)

def evaluate_state(state: Any) -> float:
    """
    Returns a scalar value for MCTS backup: higher is better for us.
    Range ~[-1, +1].

    Uses:
      - Team HP advantage (anchor)
      - Best-move damage-race advantage (tempo)
      - Best switch option (escape hatch)
      - Small status shaping (BRN/PAR)
    """
    # Terminal: if a side has no HP left, hard value
    if state.is_terminal():
        my_sum = _team_hp_sum(state.my_hp)
        opp_sum = _team_hp_sum(state.opp_hp)
        if my_sum <= 1e-9 and opp_sum <= 1e-9:
            return 0.0
        if opp_sum <= 1e-9:
            return +1.0
        if my_sum <= 1e-9:
            return -1.0
        # fallback
        return _tanh01((my_sum - opp_sum) / 2.0)

    # Patch status so race calc + heuristics see simulated BRN/PAR
    with state._patched_status(), state._patched_boosts():
        # Team HP advantage (flawed but stable, will probs come up with a better way to determine winning positions)
        my_sum = _team_hp_sum(state.my_hp)
        opp_sum = _team_hp_sum(state.opp_hp)
        hp_term = _tanh01((my_sum - opp_sum) / 2.5)  # divisor tunes sensitivity

        # Best-move race advantage (active vs active tempo)
        # Choose the move with highest heuristic score, then evaluate its race.
        best_mv = None
        best_mv_score = -1e18
        for (kind, obj) in state.legal_actions():
            if kind != "move" or obj is None:
                continue
            s = float(state.score_move_fn(obj, state.battle, state.ctx_me))
            if s > best_mv_score:
                best_mv_score = s
                best_mv = obj

        race_term = 0.0
        if best_mv is not None and state.ctx_me is not None:
            race = evaluate_race_for_move(state.battle, state.ctx_me, best_mv)

            # Convert (ttd_me - tko_opp) into a smooth score:
            # positive if we kill sooner than we die.
            # scale factor ~1.5 makes "one turn swing" meaningful but not insane.
            race_term = _tanh01((race.ttd_me - race.tko_opp) / 1.5)

        # Escape hatch: how good is our best switch (from this exact state)?
        # We only consider this when we're under real pressure (race looks bad)
        switch_term = 0.0
        if race_term < 0.0:
            best_sw = None
            best_sw_score = -1e18
            for p in state.my_team:
                if p is state.my_active:
                    continue
                if state.my_hp.get(id(p), 0.0) <= 0.0:
                    continue
                sc = float(state.score_switch_fn(p, state.battle, state.ctx_me))
                if sc > best_sw_score:
                    best_sw_score = sc
                    best_sw = p

            # Bound it so it doesn't dominate
            switch_term = _tanh01(best_sw_score / 120.0)

        boost_term = evaluate_boosts(state)

        # Status shaping (small)
        status_term = 0.0

        # Reward burning/paralyzing their active a bit
        if state.opp_status.get(id(state.opp_active)) == Status.BRN:
            status_term += 0.10
        if state.opp_status.get(id(state.opp_active)) == Status.PAR:
            status_term += 0.06

        # Penalize us being burned/paralyzed a bit
        if state.my_status.get(id(state.my_active)) == Status.BRN:
            status_term -= 0.10
        if state.my_status.get(id(state.my_active)) == Status.PAR:
            status_term -= 0.06

    # Weighted sum (anchor on HP, then tempo, then “can we safely escape”)
    value = (
        0.50 * hp_term +
        0.25 * race_term +
        0.10 * switch_term +
        0.15 * boost_term + 
        status_term
    )

    # final clamp
    return max(-1.0, min(1.0, float(value)))
