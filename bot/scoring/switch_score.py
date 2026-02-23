from typing import Any, Optional

from poke_env.battle import PokemonType, Status

from bot.model.ctx import EvalContext
from bot.scoring.helpers import hp_frac, safe_types
from bot.scoring.damage_score import estimate_damage_fraction
from bot.scoring.status_score import score_status_move
from bot.scoring.switch_belief import (
    build_opponent_belief,
    belief_free_progress_probs,
    belief_penalties_total,
) 

SPIKES_DAMAGE = {1: 1.0 / 8.0, 2: 1.0 / 6.0, 3: 1.0 / 4.0}
_SETUP_MOVE_IDS = {
    'swordsdance', 'nastyplot', 'dragondance', 'calmmind', 'bulkup',
    'quiverdance', 'shellsmash', 'bellydrum', 'shiftgear', 'agility',
    'tailglow', 'coil', 'curse', 'growth',
}
_HAZARD_MOVE_IDS = {'stealthrock', 'spikes', 'toxicspikes', 'stickyweb'}
_REMOVAL_MOVE_IDS = {'rapidspin', 'defog'}
_PIVOT_MOVE_IDS = {'uturn', 'voltswitch', 'flipturn', 'partingshot', 'chillyreception', 'teleport'}
_STATUS_MOVE_IDS = {
    # poison
    'toxic', 'poisonpowder',
    # burn
    'willowisp',
    # para
    'thunderwave', 'glare', 'stunspore',
    # sleep
    'spore', 'sleeppowder', 'hypnosis', 'sing', 'yawn', 'lovelykiss', 'grasswhistle', 'darkvoid',
}
# Chip-resilient: ability -> penalty scale (Magic Guard ignores hazards; Regenerator heals on switch)
_CHIP_RESILIENCE_SCALES = {'magicguard': 0.0, 'regenerator': 0.6}
# Anti-setup: Haze, Clear Smog, phazing, Encore, Taunt, Unaware
_ANTI_SETUP_MOVE_IDS = frozenset({
    'haze', 'clearsmog', 'roar', 'whirlwind', 'dragontail', 'encore', 'taunt',
})
_UNAWARE_ABILITY = 'unaware'


def score_switch(pokemon: Any, battle: Any, ctx: EvalContext) -> float:
    """
    Score how good switching to `pokemon` is given the current battle state.

    Args:
        pokemon: The bench Pokemon being considered for switch-in
        battle: Current battle state
        ctx: EvalContext (ctx.me = current active, ctx.opp = opponent's active)

    Returns:
        Float score (higher = better switch)
    """
    opp = ctx.opp
    me = ctx.me

    if pokemon is None or opp is None:
        return -100.0

    switch_hp = hp_frac(pokemon)
    if switch_hp <= 0.0:
        return -200.0

    score = 0.0

    # Active survival urgency
    # If our current active is about to be KO'd (especially by priority), switching is urgent.
    # This flat bonus raises all switch scores so MCTS can compare switching vs. staying correctly.
    score += _active_ko_threat(ctx, opp, battle)

    # HP factor
    # Bringing in a low-HP mon is risky; healthy mons handle pressure better
    if switch_hp < 0.25:
        score -= 25.0
    elif switch_hp < 0.5:
        score -= 8.0
    elif switch_hp > 0.75:
        score += 5.0

    # Hazard entry penalty 
    my_sc = _get_side_conditions(battle, our_side=True)
    score -= _hazard_penalty(pokemon, my_sc)

    # Hazard fraction for survival: Magic Guard ignores hazards; others take full damage
    hazard_frac = _survival_hazard_frac(pokemon, my_sc)
    effective_hp = max(0.0, switch_hp - hazard_frac)

    # Defensive matchup 
    opp_best_damage = _best_opponent_damage(opp, pokemon, battle)
    # Use expected damage (probability-weighted) for the HP penalty, max for type/survival checks.
    opp_expected_damage = _expected_opponent_damage(opp, pokemon, battle)
    score -= opp_expected_damage * 60.0

    # Resistance/immunity bonus (uses max — worst-case type check)
    if opp_best_damage < 0.05:
        score += 40.0   # Immune or 4x resistant — great switch-in
    elif opp_best_damage <= 0.15:
        score += 20.0   # 2x resistant
    elif opp_best_damage < 0.25:
        score += 8.0    # Mildly resistant

    # Survival check: will the switch-in be KO'd on entry?
    # Only penalize voluntary switches (free switches from a fainted active skip this)
    active_fainted = (me is None) or (hp_frac(me) <= 0.0)
    if not active_fainted:
        # Use effective_hp, not switch_hp
        if opp_best_damage >= effective_hp:
            score -= 50.0
        elif opp_best_damage >= effective_hp * 0.70:
            score -= 20.0

    # Offensive matchup 
    my_best_damage = _best_offensive_damage(pokemon, opp, battle)
    score += my_best_damage * 40.0

    opp_hp = hp_frac(opp)
    if my_best_damage >= opp_hp:
        score += 25.0   # Can KO immediately — forces their hand
    elif my_best_damage >= opp_hp * 0.5:
        score += 8.0    # Strong 2HKO pressure

    # Status condition on the switch-in 
    score -= _status_penalty(pokemon)

    # Belief-based free-turn weighting 
    # Build belief once; cache in ctx for multiple switch candidates per turn
    belief = _get_or_build_belief(opp, battle, ctx)
    belief_probs = belief_free_progress_probs(belief)

    # Free-turn cost: opponent's free move on the switch turn
    # When we switch voluntarily, the opponent acts freely.
    # Use belief probs to weight penalties (p_setup, p_hazards, etc.) instead of binary checks.
    if not active_fainted:
        score -= _free_turn_penalty(
            opp=opp,
            opp_hp=opp_hp,
            my_best_damage=my_best_damage,
            opp_best_damage=opp_best_damage,
            switch_in=pokemon,
            battle=battle,
            belief_probs=belief_probs,
        )

    # Belief-based risk penalties (coverage, tail risk, item swing)
    # Helmet: only when pivoting is likely (safe enough that we'd click U-turn/Flip Turn)
    has_contact_pivot = _has_contact_pivot_move(pokemon)
    likely_to_pivot = has_contact_pivot and opp_best_damage < 0.25
    score -= belief_penalties_total(
        opp, pokemon, battle, effective_hp, likely_to_pivot, belief=belief
    )

    # Passive matchup penalty
    # Taking big hits while unable to threaten back is a losing trade.
    score -= _passive_switch_penalty(opp_best_damage, my_best_damage)

    # Role preservation
    # Penalize risking role-critical mons (only remover, win-condition sweeper)
    # when the switch-in would take meaningful damage.
    score -= _role_preservation_penalty(pokemon, battle, opp_best_damage)

    # Pivot conversion bonus
    # Safe switch-in → click pivot → convert into best matchup. Gate on effective_hp (no bonus if too fragile).
    score += _pivot_conversion_bonus(pokemon, opp_best_damage, effective_hp)

    return float(score)

def _get_or_build_belief(opp: Any, battle: Any, ctx: EvalContext):
    """Build opponent belief once per turn; cache in ctx for multiple switch candidates."""
    if not hasattr(ctx, 'cache') or ctx.cache is None:
        ctx.cache = {}
    cache = ctx.cache
    try:
        t = getattr(battle, 'turn', 0)
        turn = int(t) if isinstance(t, (int, float)) else 0
    except (TypeError, ValueError):
        turn = 0
    cache_key = ('switch_opp_belief', id(opp), turn)
    if cache_key not in cache:
        try:
            fmt = getattr(battle, 'format', None)
            g = getattr(fmt, 'gen', 9) if fmt is not None else 9
            gen = int(g) if isinstance(g, (int, float)) else 9
            cache[cache_key] = build_opponent_belief(opp, gen)
        except Exception:
            cache[cache_key] = None
    return cache.get(cache_key)


def _has_contact_pivot_move(pokemon: Any) -> bool:
    """True if pokemon has contact pivot (U-turn, Flip Turn) — punished by Rocky Helmet."""
    contact_pivots = {'uturn', 'flipturn'}
    for move in (getattr(pokemon, 'moves', {}) or {}).values():
        if move and _norm_id(move) in contact_pivots:
            return True
    try:
        from bot.mcts.randbats_analyzer import get_all_possible_moves
        possible = get_all_possible_moves(pokemon)
        return bool(possible and possible & contact_pivots)
    except Exception:
        pass
    return False


def _free_turn_penalty(
    opp: Any,
    opp_hp: float,
    my_best_damage: float,
    opp_best_damage: float,
    switch_in: Any,
    battle: Any,
    belief_probs: Optional[dict] = None,
) -> float:
    """
    Penalty for handing the opponent a free action on the switch turn.

    Uses belief_probs (p_setup, p_hazards, p_status) to weight penalties when available;
    otherwise falls back to binary checks.
    """
    # If we threaten them hard, they don't get a free-value turn
    pressure = my_best_damage / max(0.01, opp_hp * 0.40)
    if pressure >= 1.0:
        return 0.0

    # Likelihood they just attack instead of doing "free progress"
    attack_preference = min(1.0, opp_best_damage / 0.40)

    # Window where free progress is likely
    free_window = (1.0 - attack_preference) * (1.0 - min(1.0, pressure))

    # Priority gate: opponent with a damaging priority move clicks it on our switch-in —
    # they're not setting up, setting hazards, or using status. Drastically shrink the window.
    if _opp_has_damaging_priority(opp):
        free_window *= 0.25

    # Pivot gate: if opponent has a pivot move they get tempo value regardless of our action.
    # We're not giving them "extra" free progress by switching — they were pivoting anyway.
    if _opp_has_pivot(opp):
        free_window *= 0.60

    penalty = 0.0

    # Setup: use p_setup when belief available, else binary
    p_setup = (belief_probs or {}).get('p_setup')
    if p_setup is None:
        p_setup = 1.0 if _opp_can_setup(opp) else 0.0
    # Priority gate: opponent with damaging priority moves is not using setup this turn
    if _opp_has_damaging_priority(opp):
        p_setup = min(p_setup or 0.0, 0.15)
    if p_setup > 0 and not _has_anti_setup(switch_in):
        hp_factor = max(0.4, opp_hp)
        penalty = max(penalty, p_setup * 18.0 * free_window * hp_factor)

    # Hazards: use p_hazards when belief available, else binary
    p_hazards = (belief_probs or {}).get('p_hazards')
    if p_hazards is None:
        p_hazards = 1.0 if _opp_can_set_hazards(opp) else 0.0
    if p_hazards > 0:
        hazard_pen = p_hazards * 10.0 * free_window
        my_sc = _get_side_conditions(battle, our_side=True)
        if _hazards_already_up_or_maxed(my_sc):
            hazard_pen *= 0.3
        penalty = max(penalty, hazard_pen)

    # Incoming status: use p_status to weight when belief available; still use status_score for severity
    p_status = (belief_probs or {}).get('p_status', 1.0)
    status_window = max(free_window, 0.25 * (1.0 - min(1.0, pressure)))
    incoming_status = _incoming_status_penalty(opp, switch_in, battle)
    if incoming_status > 0.0 and p_status > 0:
        penalty = max(penalty, p_status * incoming_status * status_window)

    # Recovery: if p_recover is high and we don't threaten 2HKO, they get free heal
    p_recover = (belief_probs or {}).get('p_recover', 0.0)
    if p_recover > 0 and my_best_damage < opp_hp * 0.5:
        penalty = max(penalty, p_recover * 8.0 * free_window)

    return float(penalty)

def _incoming_status_penalty(opp: Any, switch_in: Any, battle: Any) -> float:
    """
    How bad it is for us if opponent uses their best status move on our switch-in.
    Reuses score_status_move by flipping perspective:
      - 'me' is the opponent (they are using the move)
      - 'opp' is our switch-in (the target getting statused)
    Returns a non-negative penalty.
    """
    try:
        opp_moves = getattr(opp, "moves", {}) or {}
        if not opp_moves:
            return 0.0

        # If we're already statused, status moves won't matter much
        if getattr(switch_in, "status", None) is not None:
            return 0.0

        temp_ctx = EvalContext(me=opp, opp=switch_in, battle=battle, cache={})

        best = 0.0
        for mv in opp_moves.values():
            if mv is None:
                continue

            # score_status_move returns "value to the user of the move"
            # Here: value to the opponent, which is our penalty.
            val = float(score_status_move(mv, battle, temp_ctx))

            # status_score returns large negatives when not applicable / already statused etc.
            if val > best:
                best = val

        return max(0.0, best)

    except Exception:
        return 0.0

# Free-turn logic


def _opp_can_setup(opp: Any) -> bool:
    """True if the opponent has or likely has a setup move."""
    # Check known moves first (most reliable)
    for move in (getattr(opp, 'moves', {}) or {}).values():
        if _norm_id(move) in _SETUP_MOVE_IDS:
            return True
    # Fall back to randbats database
    try:
        from bot.mcts.randbats_analyzer import has_setup_potential
        return has_setup_potential(opp)
    except Exception:
        return False


def _opp_can_set_hazards(opp: Any) -> bool:
    """True if the opponent has a known hazard-setting move."""
    for move in (getattr(opp, 'moves', {}) or {}).values():
        if _norm_id(move) in _HAZARD_MOVE_IDS:
            return True
    return False


def _has_anti_setup(pokemon: Any) -> bool:
    """True if this mon can punish or nullify setup (Haze, Clear Smog, phaze, Encore, Taunt, Unaware)."""
    ability = str(getattr(pokemon, 'ability', '') or '').lower().replace(' ', '').replace('-', '')
    if ability == _UNAWARE_ABILITY:
        return True
    for move in (getattr(pokemon, 'moves', {}) or {}).values():
        if move and _norm_id(move) in _ANTI_SETUP_MOVE_IDS:
            return True
    try:
        from bot.mcts.randbats_analyzer import get_all_possible_moves
        possible = get_all_possible_moves(pokemon)
        if possible and possible & _ANTI_SETUP_MOVE_IDS:
            return True
    except Exception:
        pass
    return False


def _has_pivot_move(pokemon: Any) -> bool:
    """True if this mon has a pivot move (U-turn, Volt Switch, etc.)."""
    for move in (getattr(pokemon, 'moves', {}) or {}).values():
        if move and _norm_id(move) in _PIVOT_MOVE_IDS:
            return True
    try:
        from bot.mcts.randbats_analyzer import get_all_possible_moves
        possible = get_all_possible_moves(pokemon)
        return bool(possible and possible & _PIVOT_MOVE_IDS)
    except Exception:
        pass
    return False


def _opp_has_damaging_priority(opp: Any) -> bool:
    """True if opponent has a known damaging move with priority > 0 (Ice Shard, Sucker Punch, etc.)."""
    try:
        for move in (getattr(opp, 'moves', {}) or {}).values():
            if move is None:
                continue
            if int(getattr(move, 'priority', 0) or 0) > 0 and int(getattr(move, 'base_power', 0) or 0) > 0:
                return True
    except Exception:
        pass
    return False


def _opp_has_pivot(opp: Any) -> bool:
    """True if opponent has a known pivot move (U-turn, Volt Switch, Flip Turn, Parting Shot)."""
    try:
        for move in (getattr(opp, 'moves', {}) or {}).values():
            if move and _norm_id(move) in _PIVOT_MOVE_IDS:
                return True
    except Exception:
        pass
    return False


def _pivot_conversion_bonus(pokemon: Any, opp_best_damage: float, effective_hp: float) -> float:
    """
    Pivot mons gain value when they can safely switch in: safe switch-in -> click pivot -> convert into best matchup.
    Bonus proportional to how safe the entry is (low opp_best_damage). Gate: no bonus if effective_hp < 0.35 (too fragile).
    """
    if not _has_pivot_move(pokemon):
        return 0.0
    if effective_hp < 0.35:
        return 0.0  # too fragile to pivot repeatedly (hazard chip + incoming hits)
    # Safety: 0 when taking big damage, scales up as we take less
    if opp_best_damage >= 0.25:
        return 0.0
    safety = 1.0 - (opp_best_damage / 0.25)  # 1.0 when opp_best~0, 0 when opp_best=0.25
    return safety * 12.0  # up to 12 pts when completely safe


# Passive matchup penalty

def _passive_switch_penalty(opp_best_damage: float, my_best_damage: float) -> float:
    """
    Penalty for a switch-in that takes significant damage but can't threaten back.

    Taking 40–60% while doing nothing back is a losing trade — you lose HP
    and give them free progress. The penalty is proportional to the gap.
    """
    if opp_best_damage < 0.33:
        return 0.0  # Not taking meaningful damage, no passivity concern
    if my_best_damage >= 0.25:
        return 0.0  # We can at least threaten back

    passivity = opp_best_damage - my_best_damage
    return max(0.0, passivity) * 12.0


# Role preservation

def _role_preservation_penalty(
    pokemon: Any,
    battle: Any,
    opp_best_damage: float,
) -> float:
    """
    Penalty for unnecessarily risking a role-critical Pokemon.

    Protects:
    - The only hazard remover (losing it makes hazards permanent)
    - A healthy win-condition setup sweeper
    """
    if opp_best_damage < 0.20:
        return 0.0  # Safe enough regardless of role

    try:
        team = list((getattr(battle, 'team', {}) or {}).values())
        alive = [
            m for m in team
            if m is not None
            and not getattr(m, 'fainted', False)
            and hp_frac(m) > 0.0
        ]
    except Exception:
        return 0.0

    if not alive:
        return 0.0

    penalty = 0.0

    # Only hazard remover?
    removers = [m for m in alive if _has_removal(m)]
    if pokemon in removers and len(removers) == 1:
        # Losing the only remover makes hazards permanent — serious cost
        penalty += opp_best_damage * 25.0

    # Win-condition setup sweeper?
    try:
        from bot.mcts.randbats_analyzer import has_setup_potential
        is_setup = has_setup_potential(pokemon)
        if not is_setup:
            # Fallback: check known moves directly
            for mv in (getattr(pokemon, 'moves', {}) or {}).values():
                if _norm_id(mv) in _SETUP_MOVE_IDS:
                    is_setup = True
                    break
        if is_setup and hp_frac(pokemon) > 0.65:
            if opp_best_damage >= 0.35:
                penalty += 12.0
    except Exception:
        pass

    return penalty


def _has_removal(mon: Any) -> bool:
    for move in (getattr(mon, 'moves', {}) or {}).values():
        if _norm_id(move) in _REMOVAL_MOVE_IDS:
            return True
    return False


# Entry hazard helpers

def _hazards_already_up_or_maxed(sc: dict) -> bool:
    """True if our side already has hazards (or maxed: 3 spikes, 2 tspikes). Reduces free-turn hazard penalty."""
    return (
        sc.get('stealthrock', 0) > 0
        or sc.get('spikes', 0) > 0
        or sc.get('toxicspikes', 0) > 0
        or sc.get('stickyweb', 0) > 0
    )


def _get_side_conditions(battle: Any, our_side: bool) -> dict:
    """Return side conditions as a normalized lowercase string-keyed dict."""
    try:
        if our_side:
            raw = getattr(battle, 'side_conditions', {}) or {}
        else:
            raw = getattr(battle, 'opponent_side_conditions', {}) or {}
        result = {}
        for key, val in raw.items():
            name = getattr(key, 'name', str(key)).lower().replace('_', '')
            result[name] = int(val) if val else 1
        return result
    except Exception:
        return {}


def _survival_hazard_frac(pokemon: Any, sc: dict) -> float:
    """
    Hazard HP fraction that actually reduces HP for survival math.
    Magic Guard ignores entry hazards (SR/spikes are indirect); others take full damage.
    Regenerator/Leftovers don't prevent hazard damage on entry.
    """
    ability = str(getattr(pokemon, 'ability', '') or '').lower().replace(' ', '').replace('-', '')
    if ability in _CHIP_RESILIENCE_SCALES and _CHIP_RESILIENCE_SCALES[ability] == 0:
        return 0.0
    return _hazard_entry_frac(pokemon, sc)


def _hazard_penalty(pokemon: Any, sc: dict) -> float:
    """Point penalty for switching through our side's entry hazards."""
    item = str(getattr(pokemon, 'item', '') or '').lower().replace(' ', '').replace('-', '')
    if item == 'heavydutyboots':
        return 0.0

    # Chip-resilient mons: Magic Guard ignores hazards; Regenerator heals on switch; Leftovers+bulky less hurt
    scale = _chip_resilience_scale(pokemon)

    penalty = 0.0
    grounded = _is_grounded(pokemon)

    if sc.get('stealthrock', 0) > 0:
        dmg_frac = _sr_damage_frac(pokemon)
        penalty += dmg_frac * 40.0  # neutral (12.5%) → 5pts; 4x weak (50%) → 20pts

    spikes = min(3, max(0, sc.get('spikes', 0)))
    if spikes > 0 and grounded:
        penalty += SPIKES_DAMAGE[spikes] * 30.0

    tspikes = min(2, max(0, sc.get('toxicspikes', 0)))
    if tspikes > 0 and grounded and not _is_poison_type(pokemon):
        penalty += 8.0 if tspikes >= 2 else 5.0

    if sc.get('stickyweb', 0) > 0 and grounded:
        penalty += 5.0

    return penalty * scale


def _chip_resilience_scale(pokemon: Any) -> float:
    """
    Scale for hazard penalty: chip-resilient mons take less effective damage.
    - Magic Guard: ignores entry hazards entirely (0)
    - Regenerator: heals 1/3 on switch, effective cost ~2/3
    - Leftovers + bulky: modest reduction
    """
    ability = str(getattr(pokemon, 'ability', '') or '').lower().replace(' ', '').replace('-', '')
    if ability in _CHIP_RESILIENCE_SCALES:
        return _CHIP_RESILIENCE_SCALES[ability]
    try:
        from bot.mcts.randbats_analyzer import is_defensive, can_have_leftovers
        item = str(getattr(pokemon, 'item', '') or '').lower().replace(' ', '').replace('-', '')
        if (item == 'leftovers' or can_have_leftovers(pokemon)) and is_defensive(pokemon):
            return 0.85  # bulky + passive recovery
    except Exception:
        pass
    return 1.0


# Active survival & damage estimation

def _active_ko_threat(ctx: Any, opp: Any, battle: Any) -> float:
    """
    Flat bonus for any switch when our active is endangered.

    If the opponent's best move (especially a priority move) will KO our active,
    switching is a survival necessity — not just a matchup preference.
    This bonus is added to every switch candidate so MCTS can compare switching vs. staying.

    Returns 0 if active is already fainted (free switch) or safe.
    """
    me = getattr(ctx, 'me', None)
    if me is None:
        return 0.0
    active_hp = hp_frac(me)
    if active_hp <= 0.0:
        return 0.0  # fainted — score_switch won't be called (free switch path)

    opp_vs_active = _best_opponent_damage(opp, me, battle)
    has_priority = _opp_has_damaging_priority(opp)

    if opp_vs_active >= active_hp:
        # Active will be KO'd this turn
        if has_priority:
            return 35.0   # Certain KO via priority (goes first) — switch is mandatory
        return 20.0       # Likely KO if opponent faster or speed tie — strongly favor switch

    if opp_vs_active >= active_hp * 0.70:
        # In KO range — risky to stay
        if has_priority:
            return 18.0   # Priority brings us into guaranteed-KO territory
        return 8.0        # Worth considering a switch

    return 0.0


def _expected_opponent_damage(opp: Any, pokemon: Any, battle: Any) -> float:
    """
    Probability-weighted expected damage vs the pure max — avoids overpenalizing rare worst-cases.

    - Choice-locked opponent (Band/Scarf/Specs): they're stuck on one move → return max.
    - Otherwise: blend max (60%) + second-best (40%) to approximate likely damage.
    """
    known_moves = getattr(opp, 'moves', {}) or {}
    dmgs = []
    for move in known_moves.values():
        try:
            d = float(estimate_damage_fraction(move, opp, pokemon, battle))
            dmgs.append(d)
        except Exception:
            pass
    if not dmgs:
        return _type_fallback_damage(opp, pokemon)

    dmgs.sort(reverse=True)
    best = dmgs[0]

    item = str(getattr(opp, 'item', '') or '').lower().replace(' ', '').replace('-', '')
    choice_locked = item in ('choiceband', 'choicescarf', 'choicespecs')
    if choice_locked or len(dmgs) == 1:
        return best

    second = dmgs[1] if len(dmgs) > 1 else best
    return 0.60 * best + 0.40 * second


def _best_opponent_damage(opp: Any, pokemon: Any, battle: Any) -> float:
    """Max damage the opponent can deal to our switch-in (as HP fraction)."""
    known_moves = getattr(opp, 'moves', {}) or {}
    best = 0.0
    for move in known_moves.values():
        try:
            dmg = float(estimate_damage_fraction(move, opp, pokemon, battle))
            best = max(best, dmg)
        except Exception:
            pass
    if best == 0.0:
        best = _type_fallback_damage(opp, pokemon)
    return best


def _best_offensive_damage(pokemon: Any, opp: Any, battle: Any) -> float:
    """Max damage our switch-in can deal to the opponent (as HP fraction)."""
    known_moves = getattr(pokemon, 'moves', {}) or {}
    best = 0.0
    for move in known_moves.values():
        try:
            dmg = float(estimate_damage_fraction(move, pokemon, opp, battle))
            best = max(best, dmg)
        except Exception:
            pass
    return best


def _type_fallback_damage(attacker: Any, defender: Any) -> float:
    """
    Estimate damage from type matchup when moves are unknown.
    Assumes the attacker uses their best STAB type.
    """
    try:
        from poke_env.data import GenData
        type_chart = GenData.from_gen(9).type_chart
        att_types = safe_types(attacker)
        def_types = safe_types(defender)
        if not att_types or not def_types:
            return 0.35

        best_mult = 0.0
        for att_type in att_types:
            if att_type is None:
                continue
            mult = 1.0
            for def_type in def_types:
                if def_type is None:
                    continue
                mult *= PokemonType.damage_multiplier(att_type, def_type, type_chart=type_chart)
            best_mult = max(best_mult, mult)

        return 0.30 * best_mult
    except Exception:
        return 0.30


# Type / item helpers

def _sr_damage_frac(pokemon: Any) -> float:
    """Stealth Rock damage as fraction of max HP for this Pokemon."""
    try:
        from poke_env.data import GenData
        type_chart = GenData.from_gen(9).type_chart
        mult = 1.0
        for t in (getattr(pokemon, 'type_1', None), getattr(pokemon, 'type_2', None)):
            if t is not None:
                mult *= PokemonType.damage_multiplier(PokemonType.ROCK, t, type_chart=type_chart)
        return (1.0 / 8.0) * float(mult)
    except Exception:
        return 0.125

def _hazard_entry_frac(pokemon: Any, sc: dict) -> float:
    """Entry hazard damage/status as an approximate fraction of max HP on switch-in."""
    item = str(getattr(pokemon, 'item', '') or '').lower().replace(' ', '').replace('-', '')
    if item == 'heavydutyboots':
        return 0.0

    frac = 0.0
    grounded = _is_grounded(pokemon)

    if sc.get('stealthrock', 0) > 0:
        frac += _sr_damage_frac(pokemon)

    spikes = min(3, max(0, sc.get('spikes', 0)))
    if spikes > 0 and grounded:
        frac += float(SPIKES_DAMAGE[spikes])

    # Toxic Spikes / Sticky Web don’t do immediate HP damage.
    # We handle those as “status/tempo” in free-turn/status logic (points), not HP fraction.
    return max(0.0, min(1.0, frac))

def _is_grounded(pokemon: Any) -> bool:
    try:
        t1 = getattr(pokemon, 'type_1', None)
        t2 = getattr(pokemon, 'type_2', None)
        if t1 == PokemonType.FLYING or t2 == PokemonType.FLYING:
            return False
        ability = str(getattr(pokemon, 'ability', '') or '').lower().replace(' ', '').replace('-', '')
        if ability == 'levitate':
            return False
        item = str(getattr(pokemon, 'item', '') or '').lower().replace(' ', '').replace('-', '')
        if item == 'airballoon':
            return False
        return True
    except Exception:
        return True


def _is_poison_type(pokemon: Any) -> bool:
    try:
        t1 = getattr(pokemon, 'type_1', None)
        t2 = getattr(pokemon, 'type_2', None)
        return t1 == PokemonType.POISON or t2 == PokemonType.POISON
    except Exception:
        return False


def _norm_id(move: Any) -> str:
    if isinstance(move, str):
        mid = move
    else:
        mid = str(getattr(move, 'id', getattr(move, 'name', '')) or '')
    return mid.lower().replace(' ', '').replace('-', '').replace('_', '')


def _status_penalty(pokemon: Any) -> float:
    """Point penalty for an already-existing status on the switch-in."""
    status = getattr(pokemon, 'status', None)
    if status is None:
        return 0.0
    if status == Status.PAR:
        return 8.0
    if status == Status.BRN:
        return 10.0
    if status in (Status.PSN, Status.TOX):
        return 6.0
    if status in (Status.SLP, Status.FRZ):
        return 15.0
    return 0.0
