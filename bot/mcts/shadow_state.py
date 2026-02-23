from __future__ import annotations

from dataclasses import dataclass, replace, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from contextlib import contextmanager
import math
import random

from poke_env.battle import Status, MoveCategory, PokemonType, Weather, Field, SideCondition

from bot.scoring.helpers import hp_frac, is_fainted
from bot.model.opponent_model import (
    determinize_opponent,
    sample_unseen_mon,
    sample_unseen_mon_from_team_belief,
    build_team_belief,
    TeamBelief,
)

Action = Tuple[str, Any]  # ("move", Move) or ("switch", Pokemon)

EvalContext = Any
ScoreMoveFn = Callable[[Any, Any, EvalContext], float]
ScoreSwitchFn = Callable[[Any, Any, EvalContext], float]
DamageFn = Callable[[Any, Any, Any, Any], float]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _safe_gen(battle: Any) -> int:
    try:
        g = getattr(battle, 'gen', 9)
        return int(g) if g is not None else 9
    except (TypeError, ValueError):
        return 9


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, handling Mock objects and None."""
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _get_item(pokemon: Any) -> Optional[str]:
    """Get normalized item id string from a Pokemon."""
    item = getattr(pokemon, 'item', None)
    if item and isinstance(item, str) and item.strip():
        return item.lower().replace(' ', '').replace('-', '')
    return None


CHOICE_ITEMS = frozenset({'choiceband', 'choicespecs', 'choicescarf'})

SPIKES_DAMAGE = {1: 1.0 / 8.0, 2: 1.0 / 6.0, 3: 1.0 / 4.0}


def _is_grounded(pokemon: Any) -> bool:
    """Check if a Pokemon is grounded (not Flying-type). Ignores Levitate for simplicity."""
    try:
        t1 = getattr(pokemon, 'type_1', None)
        t2 = getattr(pokemon, 'type_2', None)
        return not (t1 == PokemonType.FLYING or t2 == PokemonType.FLYING)
    except Exception:
        return True


def _is_poison_type(pokemon: Any) -> bool:
    try:
        t1 = getattr(pokemon, 'type_1', None)
        t2 = getattr(pokemon, 'type_2', None)
        return t1 == PokemonType.POISON or t2 == PokemonType.POISON
    except Exception:
        return False


def _rock_effectiveness(pokemon: Any) -> float:
    """Rock-type damage multiplier against a Pokemon's types."""
    try:
        from poke_env.data import GenData
        type_chart = GenData.from_gen(9).type_chart
        mult = 1.0
        for t in [getattr(pokemon, 'type_1', None), getattr(pokemon, 'type_2', None)]:
            if t is not None:
                mult *= PokemonType.damage_multiplier(PokemonType.ROCK, t, type_chart=type_chart)
        return float(mult)
    except Exception:
        return 1.0


def _apply_hazards_on_entry(
    pokemon: Any,
    side_conditions: Dict[str, int],
    current_status: Optional[Status],
) -> Tuple[float, Optional[Status], Optional[Dict[str, int]], Dict[str, int]]:
    """
    Calculate hazard effects when a Pokemon switches in.

    Returns:
        (damage_frac, new_status_or_None, spe_boost_change_or_None, updated_side_conditions)
    """
    # Heavy-Duty Boots negates all hazards
    if _get_item(pokemon) == 'heavydutyboots':
        return (0.0, None, None, side_conditions)

    dmg = 0.0
    inflict_status: Optional[Status] = None
    spe_change: Optional[Dict[str, int]] = None
    updated_sc = dict(side_conditions)
    grounded = _is_grounded(pokemon)

    # Stealth Rock: 1/8 * rock effectiveness
    if updated_sc.get('stealthrock', 0) > 0:
        dmg += (1.0 / 8.0) * _rock_effectiveness(pokemon)

    # Spikes: grounded only
    spikes_layers = min(3, max(0, updated_sc.get('spikes', 0)))
    if spikes_layers > 0 and grounded:
        dmg += SPIKES_DAMAGE.get(spikes_layers, 0.0)

    # Toxic Spikes: grounded only
    tspikes_layers = min(2, max(0, updated_sc.get('toxicspikes', 0)))
    if tspikes_layers > 0 and grounded:
        if _is_poison_type(pokemon):
            # Grounded Poison-type absorbs Toxic Spikes
            updated_sc['toxicspikes'] = 0
        elif current_status is None:
            # Only inflict if not already statused
            if tspikes_layers >= 2:
                inflict_status = Status.TOX
            else:
                inflict_status = Status.PSN

    # Sticky Web: grounded only, -1 spe
    if updated_sc.get('stickyweb', 0) > 0 and grounded:
        spe_change = {'spe': -1}

    return (dmg, inflict_status, spe_change, updated_sc)


def _side_conditions_to_dict(sc: Any) -> Dict[str, int]:
    """Convert battle.side_conditions (Dict[SideCondition, int]) to Dict[str, int]."""
    if not sc:
        return {}
    result = {}
    for key, val in sc.items():
        # key might be a SideCondition enum or a string
        name = getattr(key, 'name', str(key)).lower().replace('_', '')
        result[name] = int(val) if val else 1
    return result


HAZARD_REMOVAL_OWN_SIDE = frozenset({'rapidspin', 'tidyup', 'mortalspin'})
HAZARD_REMOVAL_BOTH_SIDES = frozenset({'defog'})
HAZARD_KEYS = frozenset({'stealthrock', 'spikes', 'toxicspikes', 'stickyweb'})

PROTECT_MOVES = frozenset({
    'protect', 'detect', 'kingsshield', 'banefulbunker',
    'spikyshield', 'silktrap', 'obstruct',
})

TIMED_SIDE_CONDITIONS = frozenset({
    'reflect', 'lightscreen', 'auroraveil', 'tailwind',
})

# Build mapping from lowercase-no-underscore name -> SideCondition enum
_SC_NAME_TO_ENUM: Dict[str, SideCondition] = {}
for _sc in SideCondition:
    _SC_NAME_TO_ENUM[_sc.name.lower().replace('_', '')] = _sc


def _dict_to_side_conditions(sc_dict: Dict[str, int]) -> Dict[SideCondition, int]:
    """Convert our Dict[str, int] back to Dict[SideCondition, int] for battle patching."""
    result = {}
    for name, val in sc_dict.items():
        enum_val = _SC_NAME_TO_ENUM.get(name)
        if enum_val is not None:
            result[enum_val] = val
    return result


def is_pivot_move(move: Any) -> bool:
    mid = str(getattr(move, "id", "") or getattr(move, "name", "")).lower().replace(" ", "")
    return mid in {
        "voltswitch", "uturn", "flipturn", "partingshot",
        "teleport", "chillyreception", "batonpass", "shedtail",
    }


def move_priority(move: Any) -> int:
    try:
        return int(getattr(move, "priority", 0) or 0)
    except Exception:
        return 0


def _check_flinch(action: Any, rng: random.Random) -> bool:
    """Check if a move action causes flinch. Only meaningful for first-mover."""
    if not isinstance(action, tuple) or action[0] != "move" or action[1] is None:
        return False
    secondary = getattr(action[1], 'secondary', None)
    if not secondary or not isinstance(secondary, list):
        return False
    for sec in secondary:
        if isinstance(sec, dict) and sec.get('volatileStatus') == 'flinch':
            chance = sec.get('chance', 0) / 100.0
            return rng.random() < chance
    return False

def status_infliction(move: Any) -> Optional[Tuple[Status, float]]:
    """
    Returns (status, probability) for the move's major status effect.
    
    First checks move.status for guaranteed status (e.g., Will-O-Wisp).
    Then checks move.secondary for secondary effect status (e.g., Sacred Fire).
    """
    
    # Check for guaranteed status (move.status)
    status = getattr(move, 'status', None)
    if status:
        if isinstance(status, Status):
            return (status, 1.0)
        elif isinstance(status, str):
            try:
                return (Status[status.upper()], 1.0)
            except:
                pass
    
    # Check for secondary effect status (move.secondary)
    secondary = getattr(move, 'secondary', None)
    if secondary and isinstance(secondary, list):
        for sec in secondary:
            if isinstance(sec, dict):
                status_str = sec.get('status')
                chance = sec.get('chance', 0)
                
                if status_str and chance > 0:
                    try:
                        status_enum = Status[status_str.upper()]
                        return (status_enum, chance / 100.0)
                    except:
                        pass
    
    return None

def get_move_boosts(move: Any) -> Optional[Tuple[Optional[Dict[str, int]], Optional[Dict[str, int]], float]]:
    """
    Extract stat boost changes from a move.

    Returns tuple of (self_boosts, target_boosts, chance) or None:
    - self_boosts: boosts applied to user (e.g., Swords Dance: {'atk': 2})
    - target_boosts: boosts applied to opponent (e.g., Charm: {'atk': -2})
    - chance: probability of applying (1.0 for guaranteed, 0.2 for 20%, etc.)

    poke-env stores boosts in three different places:
    - move.boosts: status moves; who is affected is given by move.target ('self' = user, else = target)
    - move.self_boost: guaranteed self-boosts on damaging moves (Close Combat, Draco Meteor)
    - move.secondary[i]['self']['boosts']: secondary self-boosts (Rapid Spin, Trailblaze)
    - move.secondary[i]['boosts']: secondary target-boosts (Crunch def drop)
    """
    # 1. Status moves with direct boosts (Swords Dance, Dragon Dance, Charm, Screech, etc.)
    boosts = getattr(move, 'boosts', None)
    if boosts:
        target_who = str(getattr(move, 'target', None) or '').lower() if move else ''
        if target_who == 'self':
            return (boosts, None, 1.0)   # user (e.g. Swords Dance, Dragon Dance)
        # target is foe / normal / etc. → effect applies to the move's target (opponent when we use it)
        return (None, boosts, 1.0)   # target (e.g. Charm, Screech, Metal Sound)

    # 2. Guaranteed self-boost on damaging moves (Close Combat, Draco Meteor, Overheat, Superpower)
    #    poke-env uses move.self_boost for these
    self_boost = getattr(move, 'self_boost', None)
    if self_boost and isinstance(self_boost, dict):
        return (self_boost, None, 1.0)

    # 3. Fallback: check move.self dict (some poke-env versions)
    self_data = getattr(move, 'self', None)
    if self_data and isinstance(self_data, dict):
        self_boosts = self_data.get('boosts', None)
        if self_boosts:
            return (self_boosts, None, 1.0)

    # 4. Secondary effects (chance-based or guaranteed-as-secondary)
    secondary = getattr(move, 'secondary', None)
    if secondary:
        sec_list = secondary if isinstance(secondary, list) else [secondary]
        for sec in sec_list:
            if not isinstance(sec, dict):
                continue
            chance = sec.get('chance', 100) / 100.0

            # 4a. Secondary self-boosts (Rapid Spin +1 Spe, Trailblaze +1 Spe, Power-Up Punch +1 Atk)
            #     Format: {'chance': 100, 'self': {'boosts': {'spe': 1}}}
            self_sec = sec.get('self')
            if isinstance(self_sec, dict):
                self_sec_boosts = self_sec.get('boosts')
                if self_sec_boosts:
                    return (self_sec_boosts, None, chance)

            # 4b. Secondary target-boosts (Crunch -1 Def, Lava Plume 30% burn has no boosts)
            #     Format: {'chance': 20, 'boosts': {'def': -1}}
            sec_boosts = sec.get('boosts', None)
            if sec_boosts:
                return (None, sec_boosts, chance)

    return None


def safe_accuracy(move: Any) -> float:
    try:
        acc = float(getattr(move, "accuracy", 1.0) or 1.0)
    except Exception:
        acc = 1.0
    return max(0.0, min(1.0, acc))


def apply_damage_with_crit(
    *,
    dmg_frac: float,
    rng: random.Random,
    model_crit: bool,
    crit_chance: float,
    crit_multiplier: float,
    forced_crit: Optional[bool] = None,
) -> Tuple[float, bool]:
    dmg_frac = float(dmg_frac)
    if not math.isfinite(dmg_frac):
        dmg_frac = 0.0
    dmg_frac = max(0.0, dmg_frac)

    if not model_crit:
        return dmg_frac, False
    
    # Check for forced crit outcome (for hybrid expansion)
    if forced_crit is not None:
        if forced_crit:
            dmg_frac *= float(crit_multiplier)
            return dmg_frac, True
        else:
            return dmg_frac, False

    # Normal crit sampling
    if rng.random() < float(crit_chance):
        dmg_frac *= float(crit_multiplier)
        return dmg_frac, True

    return dmg_frac, False


def apply_expected_damage(
    state: "ShadowState",
    move: Any,
    attacker: Any,
    defender: Any,
    defender_hp: float,
    *,
    rng: random.Random,
    hit: bool,
) -> Tuple[float, bool]:
    """
    Use dmg_fn (estimate_damage_fraction) to get fraction of defender HP removed.
    We patch statuses during the call so damage calc sees BRN/PAR.

    Stochastic handling:
      - if hit=False => no damage
      - if model_crit => possible crit multiplier
    """
    if move is None or not hit:
        return float(defender_hp), False

    # Non-damaging moves should never deal damage in the shadow sim.
    # (Boosting / status effects are handled elsewhere.)
    if float(getattr(move, 'base_power', 0) or 0) <= 0:
        return float(defender_hp), False

    try:
        with state._patched_status(), state._patched_boosts(), state._patched_fields():
            dmg_frac = float(state.dmg_fn(move, attacker, defender, state.battle))
    except Exception:
        dmg_frac = 0.0
    
    # Check for forced crit (for hybrid expansion)
    forced_crit = getattr(state, '_forced_crit', None)

    dmg_frac, did_crit = apply_damage_with_crit(
        dmg_frac=dmg_frac,
        rng=rng,
        model_crit=state.model_crit,
        crit_chance=state.crit_chance,
        crit_multiplier=state.crit_multiplier,
        forced_crit=forced_crit,
    )

    return max(0.0, float(defender_hp) - float(dmg_frac)), did_crit


@dataclass(frozen=True)
class ShadowState:
    battle: Any
    ctx_me: EvalContext
    ctx_opp: EvalContext

    my_active: Any
    opp_active: Any

    my_team: List[Any]
    opp_team: List[Any]

    my_hp: Dict[int, float]
    opp_hp: Dict[int, float]

    my_status: Dict[int, Optional[Status]]
    opp_status: Dict[int, Optional[Status]]

    my_boosts: Dict[int, Dict[str, int]] = field(default_factory=dict)
    opp_boosts: Dict[int, Dict[str, int]] = field(default_factory=dict)

    my_toxic_counter: int = 0
    opp_toxic_counter: int = 0

    my_choice_lock: Optional[str] = None   # move id we're locked into
    opp_choice_lock: Optional[str] = None  # move id opponent is locked into

    # Volatile statuses (per-pokemon, cleared on switch)
    # Keys: 'sleep_turns' (int), 'confusion_turns' (int)
    my_volatiles: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    opp_volatiles: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # Protect consecutive-use counter (for diminishing success rate)
    my_protect_count: int = 0
    opp_protect_count: int = 0

    my_side_conditions: Dict[str, int] = field(default_factory=dict)
    opp_side_conditions: Dict[str, int] = field(default_factory=dict)

    # Weather/fields tracking (mirrors battle._weather and battle._fields format)
    shadow_weather: Dict[Any, int] = field(default_factory=dict)
    shadow_fields: Dict[Any, int] = field(default_factory=dict)

    opp_beliefs: Optional[Dict[int, Any]] = None       # pokemon_id -> OpponentBelief
    opp_move_pools: Optional[Dict[int, Dict[str, Any]]] = None  # pokemon_id -> {move_id: Move}
    opp_slots: List[Optional[Any]] = field(default_factory=lambda: [None] * 6)  # 6 slots, None = unseen
    opp_active_idx: int = 0
    team_belief: Optional[Any] = None  # TeamBelief over unseen species (without-replacement per branch)
    gen: int = 9

    ply: int = 0

    # Set by step() when our active fainted this turn and the auto-switch already replaced them.
    # evaluate_state() in eval.py returns this directly so MCTS correctly penalises the KO
    # rather than seeing the deceptively-normal evaluation of the replacement Pokemon.
    pre_autoswitch_eval: Optional[float] = None

    score_move_fn: Optional[ScoreMoveFn] = None
    score_switch_fn: Optional[ScoreSwitchFn] = None
    dmg_fn: Optional[DamageFn] = None

    opp_tau: float = 8.0
    status_threshold: float = 0.30

    model_miss: bool = True
    model_crit: bool = True
    crit_chance: float = 1.0 / 24.0
    crit_multiplier: float = 1.5

    events: Tuple[str, ...] = ()
    debug: bool = False
    
    # Cache for expensive operations (role weights, damage, etc.)
    cache: Optional[Any] = None

    @staticmethod
    def from_battle(
        *,
        battle: Any,
        ctx_me: EvalContext,
        ctx_opp: EvalContext,
        score_move_fn: ScoreMoveFn,
        score_switch_fn: ScoreSwitchFn,
        dmg_fn: DamageFn,
        opp_tau: float = 8.0,
        status_threshold: float = 0.30,
        model_miss: bool = True,
        model_crit: bool = True,
        crit_chance: float = 1.0 / 24.0,
        crit_multiplier: float = 1.5,
        debug: bool = False,
        cache: Optional[Any] = None,
        opp_beliefs: Optional[Dict[int, Any]] = None,
        opp_move_pools: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> "ShadowState":
        me = ctx_me.me
        opp = ctx_me.opp

        my_team = [p for p in getattr(battle, "team", {}).values() if p]
        opp_team = [p for p in getattr(battle, "opponent_team", {}).values() if p]

        # Build opp_slots: 6 slots, known mons in 0..N-1, rest None
        opp_slots: List[Optional[Any]] = [None] * 6
        for i, p in enumerate(opp_team):
            if i < 6:
                opp_slots[i] = p
        opp_active_idx = 0
        for i, p in enumerate(opp_team):
            if p is opp:
                opp_active_idx = i
                break

        revealed_species = set()
        for p in opp_team:
            s = getattr(p, "species", None) or getattr(p, "base_species", None)
            if s:
                revealed_species.add(str(s).strip())
        team_belief = build_team_belief(_safe_gen(battle), revealed_species)

        my_hp = {id(p): _clamp01(hp_frac(p)) for p in my_team}
        opp_hp = {id(p): _clamp01(hp_frac(p)) for p in opp_team}

        my_status = {id(p): getattr(p, "status", None) for p in my_team}
        opp_status = {id(p): getattr(p, "status", None) for p in opp_team}

        my_boosts = {}
        for p in my_team:
            boosts = getattr(p, 'boosts', {})
            my_boosts[id(p)] = {
                'atk': boosts.get('atk', 0),
                'def': boosts.get('def', 0),
                'spa': boosts.get('spa', 0),
                'spd': boosts.get('spd', 0),
                'spe': boosts.get('spe', 0),
                'accuracy': boosts.get('accuracy', 0),
                'evasion': boosts.get('evasion', 0),
            }
        
        opp_boosts = {}
        for p in opp_team:
            boosts = getattr(p, 'boosts', {})
            opp_boosts[id(p)] = {
                'atk': boosts.get('atk', 0),
                'def': boosts.get('def', 0),
                'spa': boosts.get('spa', 0),
                'spd': boosts.get('spd', 0),
                'spe': boosts.get('spe', 0),
                'accuracy': boosts.get('accuracy', 0),
                'evasion': boosts.get('evasion', 0),
            }

        return ShadowState(
            battle=battle,
            ctx_me=ctx_me,
            ctx_opp=ctx_opp,
            my_active=me,
            opp_active=opp,
            my_team=my_team,
            opp_team=opp_team,
            opp_slots=opp_slots,
            opp_active_idx=opp_active_idx,
            team_belief=team_belief,
            my_hp=my_hp,
            opp_hp=opp_hp,
            my_status=my_status,
            opp_status=opp_status,
            my_boosts=my_boosts,
            opp_boosts=opp_boosts,
            my_side_conditions=_side_conditions_to_dict(
                getattr(battle, 'side_conditions', None)),
            opp_side_conditions=_side_conditions_to_dict(
                getattr(battle, 'opponent_side_conditions', None)),
            shadow_weather=dict(getattr(battle, 'weather', {}) or {}),
            shadow_fields=dict(getattr(battle, 'fields', {}) or {}),
            score_move_fn=score_move_fn,
            score_switch_fn=score_switch_fn,
            dmg_fn=dmg_fn,
            opp_tau=opp_tau,
            status_threshold=status_threshold,
            model_miss=model_miss,
            model_crit=model_crit,
            crit_chance=crit_chance,
            crit_multiplier=crit_multiplier,
            debug=bool(debug),
            cache=cache,
            opp_beliefs=opp_beliefs,
            opp_move_pools=opp_move_pools,
            gen=_safe_gen(battle),
        )

    @contextmanager
    def _patched_status(self):
        my_p, opp_p = self.my_active, self.opp_active
        old_my = getattr(my_p, "status", None) if not isinstance(my_p, tuple) else None
        old_opp = getattr(opp_p, "status", None) if not isinstance(opp_p, tuple) else None

        try:
            if not isinstance(my_p, tuple) and hasattr(my_p, "status"):
                my_p.status = self.my_status.get(id(my_p), old_my)
            if not isinstance(opp_p, tuple) and hasattr(opp_p, "status"):
                opp_p.status = self.opp_status.get(id(opp_p), old_opp)
            yield
        finally:
            if not isinstance(my_p, tuple) and hasattr(my_p, "status"):
                my_p.status = old_my
            if not isinstance(opp_p, tuple) and hasattr(opp_p, "status"):
                opp_p.status = old_opp

    @contextmanager
    def _patched_boosts(self):
        """
        Temporarily patch Pokemon.boosts with simulated boost values.
        
        This allows damage calculation and other functions to use the correct
        boost values from the simulation state.
        """
        
        # Store original boosts
        original_my = {}
        original_opp = {}
        
        # Patch my team's boosts
        for p in self.my_team:
            p_id = id(p)
            if p_id in self.my_boosts:
                original_my[p_id] = getattr(p, 'boosts', {}).copy()
                p.boosts = self.my_boosts[p_id].copy()
        
        # Patch opponent team's boosts
        for p in self.opp_team:
            p_id = id(p)
            if p_id in self.opp_boosts:
                original_opp[p_id] = getattr(p, 'boosts', {}).copy()
                p.boosts = self.opp_boosts[p_id].copy()
        
        try:
            yield
        finally:
            # Restore original boosts
            for p in self.my_team:
                p_id = id(p)
                if p_id in original_my:
                    p.boosts = original_my[p_id]
            
            for p in self.opp_team:
                p_id = id(p)
                if p_id in original_opp:
                    p.boosts = original_opp[p_id]
    
    @contextmanager
    def _patched_fields(self):
        """Temporarily patch battle weather/fields/side_conditions from shadow state.

        Works with both real Battle objects (private _attrs) and Mock battles
        (public attrs set directly).
        """
        battle = self.battle

        # Detect real Battle (has _weather as a dict) vs Mock (only public weather)
        use_private = isinstance(getattr(battle, '_weather', None), dict)

        if use_private:
            old_weather = battle._weather
            old_fields = battle._fields
            old_sc = battle._side_conditions
            old_opp_sc = battle._opponent_side_conditions
            try:
                battle._weather = dict(self.shadow_weather)
                battle._fields = dict(self.shadow_fields)
                battle._side_conditions = _dict_to_side_conditions(self.my_side_conditions)
                battle._opponent_side_conditions = _dict_to_side_conditions(self.opp_side_conditions)
                yield
            finally:
                battle._weather = old_weather
                battle._fields = old_fields
                battle._side_conditions = old_sc
                battle._opponent_side_conditions = old_opp_sc
        else:
            # Mock / simple object: patch public attributes
            old_weather = getattr(battle, 'weather', {})
            old_fields = getattr(battle, 'fields', {})
            old_sc = getattr(battle, 'side_conditions', {})
            old_opp_sc = getattr(battle, 'opponent_side_conditions', {})
            try:
                battle.weather = dict(self.shadow_weather)
                battle.fields = dict(self.shadow_fields)
                battle.side_conditions = _dict_to_side_conditions(self.my_side_conditions)
                battle.opponent_side_conditions = _dict_to_side_conditions(self.opp_side_conditions)
                yield
            finally:
                battle.weather = old_weather
                battle.fields = old_fields
                battle.side_conditions = old_sc
                battle.opponent_side_conditions = old_opp_sc

    def with_forced_outcome(self, hit: Optional[bool] = None, crit: Optional[bool] = None) -> "ShadowState":
        """
        Return a copy of this state with forced outcomes for the next move.
        Used by hybrid expansion to create deterministic branches.
        
        Args:
            hit: If set, force the next move to hit (True) or miss (False)
            crit: If set, force the next move to crit (True) or not crit (False)
        
        Returns:
            New ShadowState with forced outcomes set
        """
        new_state = replace(self)
        
        # Use object attributes to avoid issues with frozen dataclass
        if hit is not None:
            object.__setattr__(new_state, '_forced_hit', hit)
        if crit is not None:
            object.__setattr__(new_state, '_forced_crit', crit)
        
        return new_state

    def my_active_hp(self) -> float:
        return float(self.my_hp.get(id(self.my_active), 1.0))

    def opp_active_hp(self) -> float:
        return float(self.opp_hp.get(id(self.opp_active), 1.0))

    def my_active_status(self) -> Optional[Status]:
        return self.my_status.get(id(self.my_active))

    def opp_active_status(self) -> Optional[Status]:
        return self.opp_status.get(id(self.opp_active))

    def is_terminal(self) -> bool:
        return all(v <= 0.0 for v in self.my_hp.values()) or all(v <= 0.0 for v in self.opp_hp.values())

    # Speed stage multipliers (same as game: 0->1, +1->1.5, -1->2/3, etc.)
    _SPEED_STAGE_MULT = {
        -6: 2.0 / 8, -5: 2.0 / 7, -4: 2.0 / 6, -3: 2.0 / 5, -2: 2.0 / 4, -1: 2.0 / 3,
        0: 1.0, 1: 1.5, 2: 2.0, 3: 2.5, 4: 3.0, 5: 3.5, 6: 4.0,
    }

    def _effective_speed(self, p: Any, side: str) -> float:
        """Effective speed for turn order: base stat, status, item, and simulated boost stages."""
        s = float((p.stats or {}).get('spe', 100))
        st = (self.my_status if side == "me" else self.opp_status).get(id(p))
        if st == Status.PAR:
            s *= 0.5
        item = _get_item(p)
        if item == 'choicescarf':
            s *= 1.5
        # Apply speed boost from simulation state so turn order is correct after Dragon Dance etc.
        boosts = (self.my_boosts if side == "me" else self.opp_boosts).get(id(p), {})
        spe_stage = boosts.get('spe', 0)
        s *= self._SPEED_STAGE_MULT.get(max(-6, min(6, spe_stage)), 1.0)
        # Tailwind: 2x speed
        sc = self.my_side_conditions if side == "me" else self.opp_side_conditions
        if 'tailwind' in sc:
            s *= 2.0
        return s

    def _order_for_turn(self, my_action: Action, opp_action: Action, rng: random.Random) -> int:
        """
        Determine action order for the turn.
        
        Order priority:
        1. Switches always go before moves
        2. Between moves: higher priority first, then speed, then random
        3. Between switches: faster Pokemon switches first (simultaneous switches)
        """
        my_is_switch = my_action[0] == "switch"
        opp_is_switch = opp_action[0] == "switch"
        
        # If one is switching and the other isn't, switch goes first
        if my_is_switch and not opp_is_switch:
            return +1  # We switch first
        if opp_is_switch and not my_is_switch:
            return -1  # Opponent switches first
        
        # If both switching, faster Pokemon switches first (or random on speed tie)
        if my_is_switch and opp_is_switch:
            ms = self._effective_speed(self.my_active, "me")
            os = self._effective_speed(self.opp_active, "opp")
            if ms != os:
                return +1 if ms > os else -1
            return +1 if rng.random() < 0.5 else -1
        
        # Both using moves - check priority, then speed
        mp = move_priority(my_action[1]) if my_action[0] == "move" else 0
        op = move_priority(opp_action[1]) if opp_action[0] == "move" else 0

        if mp != op:
            return +1 if mp > op else -1

        ms = self._effective_speed(self.my_active, "me")
        os = self._effective_speed(self.opp_active, "opp")

        # Trick Room reverses speed ordering for moves
        trick_room = Field.TRICK_ROOM in self.shadow_fields

        if ms != os:
            if trick_room:
                return +1 if ms < os else -1  # slower moves first
            return +1 if ms > os else -1

        # speed tie
        result = +1 if rng.random() < 0.5 else -1
        return result

    def legal_actions(self) -> List[Action]:
        """All legal actions for *our* side from this ShadowState.

        Key rule: if our active is fainted, the only legal actions are switches
        to non-fainted teammates. This avoids impossible "move while fainted"
        transitions and keeps logs/rollouts consistent.
        """
        actions: List[Action] = []

        # Forced replacement: a fainted active cannot act.
        if self.my_hp.get(id(self.my_active), 1.0) <= 0.0:
            for p in self.my_team:
                if p is self.my_active:
                    continue
                if is_fainted(p) or self.my_hp.get(id(p), 1.0) <= 0.0:
                    continue
                actions.append(("switch", p))
            return actions or [("move", None)]

        # Moves (turn-0 can use battle.available_moves if present, else fall back)
        if self.ply == 0 and getattr(self.battle, "available_moves", None):
            for m in self.battle.available_moves:
                actions.append(("move", m))
        else:
            for m in (getattr(self.my_active, "moves", None) or {}).values():
                actions.append(("move", m))

        # Choice lock: if locked, only allow the locked move
        if self.my_choice_lock and _get_item(self.my_active) in CHOICE_ITEMS:
            locked = [a for a in actions if a[0] == "move"
                      and str(getattr(a[1], 'id', '')) == self.my_choice_lock]
            if locked:
                actions = locked

        # Sleep-aware move filtering
        if self.my_status.get(id(self.my_active)) == Status.SLP:
            my_vol = self.my_volatiles.get(id(self.my_active), {})
            sleep_turns = my_vol.get('sleep_turns', 0)
            if sleep_turns > 1:
                # Will still be asleep after decrement — only sleep-usable moves
                sleep_moves = [a for a in actions if a[0] == "move"
                               and getattr(a[1], 'sleep_usable', False)]
                if sleep_moves:
                    actions = sleep_moves
            elif sleep_turns == 1:
                # Guaranteed wake — use normal moves, not Sleep Talk
                normal_moves = [a for a in actions if a[0] == "move"
                                and not getattr(a[1], 'sleep_usable', False)]
                if normal_moves:
                    actions = normal_moves

        # Switches
        for p in self.my_team:
            if p is self.my_active:
                continue
            if is_fainted(p) or self.my_hp.get(id(p), 1.0) <= 0.0:
                continue
            actions.append(("switch", p))

        return actions or [("move", None)]


    def legal_actions_opp(self) -> List[Action]:
        """All legal actions for the opponent side from this ShadowState."""
        actions: List[Action] = []

        # Forced replacement: a fainted active cannot act.
        if self.opp_hp.get(id(self.opp_active), 1.0) <= 0.0:
            for p in self.opp_team:
                if p is self.opp_active:
                    continue
                if is_fainted(p) or self.opp_hp.get(id(p), 1.0) <= 0.0:
                    continue
                actions.append(("switch", p))
            opp_slots = getattr(self, "opp_slots", None) or [None] * 6
            opp_active_idx = getattr(self, "opp_active_idx", 0)
            tb = getattr(self, "team_belief", None)
            if tb is not None and tb.has_mass():
                for idx in range(min(6, len(opp_slots))):
                    if opp_slots[idx] is None and idx != opp_active_idx:
                        actions.append(("switch_unknown", idx))
            return actions or [("move", None)]

        for m in (getattr(self.opp_active, "moves", None) or {}).values():
            actions.append(("move", m))

        # Choice lock: if locked, only allow the locked move
        if self.opp_choice_lock and _get_item(self.opp_active) in CHOICE_ITEMS:
            locked = [a for a in actions if a[0] == "move"
                      and str(getattr(a[1], 'id', '')) == self.opp_choice_lock]
            if locked:
                actions = locked

        # Sleep-aware move filtering
        if self.opp_status.get(id(self.opp_active)) == Status.SLP:
            opp_vol = self.opp_volatiles.get(id(self.opp_active), {})
            sleep_turns = opp_vol.get('sleep_turns', 0)
            if sleep_turns > 1:
                sleep_moves = [a for a in actions if a[0] == "move"
                               and getattr(a[1], 'sleep_usable', False)]
                if sleep_moves:
                    actions = sleep_moves
            elif sleep_turns == 1:
                normal_moves = [a for a in actions if a[0] == "move"
                                and not getattr(a[1], 'sleep_usable', False)]
                if normal_moves:
                    actions = normal_moves

        for p in self.opp_team:
            if p is self.opp_active:
                continue
            if is_fainted(p) or self.opp_hp.get(id(p), 1.0) <= 0.0:
                continue
            actions.append(("switch", p))

        opp_slots = getattr(self, "opp_slots", None) or [None] * 6
        opp_active_idx = getattr(self, "opp_active_idx", 0)
        tb = getattr(self, "team_belief", None)
        if tb is not None and tb.has_mass():
            for idx in range(min(6, len(opp_slots))):
                if opp_slots[idx] is None and idx != opp_active_idx:
                    actions.append(("switch_unknown", idx))

        return actions or [("move", None)]

    def choose_opp_action(self, rng: random.Random) -> Action:
        opp_id = id(self.opp_active)
        belief = (self.opp_beliefs or {}).get(opp_id)
        move_pool = (self.opp_move_pools or {}).get(opp_id)

        if belief and move_pool:
            actions = self._belief_actions_opp(belief, move_pool, rng)
        else:
            actions = self.legal_actions_opp()

        # Peek-sample one unseen mon to score all switch_unknown actions (don't commit)
        temp_unknown_mon: Optional[Any] = None
        if any(k == "switch_unknown" for k, _ in actions):
            tb = getattr(self, "team_belief", None)
            if tb is not None and tb.has_mass():
                temp_unknown_mon, _ = sample_unseen_mon_from_team_belief(tb, self.gen, rng)
            if temp_unknown_mon is None:
                known_ids = set()
                for p in self.opp_team:
                    s = getattr(p, "species", None) or getattr(p, "base_species", None)
                    if s:
                        known_ids.add(str(s).strip())
                temp_unknown_mon = sample_unseen_mon(self.gen, known_ids, rng)

        scores: List[float] = []
        with self._patched_status(), self._patched_boosts():
            for k, o in actions:
                if k == "move":
                    scores.append(float(self.score_move_fn(o, self.battle, self.ctx_opp)))
                elif k == "switch_unknown":
                    if temp_unknown_mon is not None:
                        scores.append(float(self.score_switch_fn(temp_unknown_mon, self.battle, self.ctx_opp)))
                    else:
                        scores.append(0.0)
                else:
                    scores.append(float(self.score_switch_fn(o, self.battle, self.ctx_opp)))

        return self._softmax(actions, scores, self.opp_tau, rng)

    def _belief_actions_opp(self, belief: Any, move_pool: Dict[str, Any],
                            rng: random.Random) -> List[Action]:
        """Build opponent action list using belief-sampled moves."""

        # Forced replacement: fainted active can only switch
        if self.opp_hp.get(id(self.opp_active), 1.0) <= 0.0:
            return self.legal_actions_opp()

        # Determinize: sample a role and 4-move subset
        det = determinize_opponent(belief, rng)

        # Build action list from sampled moves
        seen_ids: set = set()
        actions: List[Action] = []

        for mid in det.moves4:
            mv = move_pool.get(mid)
            if mv is not None and mid not in seen_ids:
                seen_ids.add(mid)
                actions.append(("move", mv))

        # Also include revealed moves (always available, might not be in sampled set)
        for m in (getattr(self.opp_active, "moves", None) or {}).values():
            mid = str(getattr(m, 'id', ''))
            if mid not in seen_ids:
                seen_ids.add(mid)
                actions.append(("move", m))

        # Apply choice lock filter
        if self.opp_choice_lock and _get_item(self.opp_active) in CHOICE_ITEMS:
            locked = [a for a in actions if a[0] == "move"
                      and str(getattr(a[1], 'id', '')) == self.opp_choice_lock]
            if locked:
                actions = locked

        # Add switches
        for p in self.opp_team:
            if p is self.opp_active:
                continue
            if is_fainted(p) or self.opp_hp.get(id(p), 1.0) <= 0.0:
                continue
            actions.append(("switch", p))

        opp_slots = getattr(self, "opp_slots", None) or [None] * 6
        opp_active_idx = getattr(self, "opp_active_idx", 0)
        tb = getattr(self, "team_belief", None)
        if tb is not None and tb.has_mass():
            for idx in range(min(6, len(opp_slots))):
                if opp_slots[idx] is None and idx != opp_active_idx:
                    actions.append(("switch_unknown", idx))

        return actions or [("move", None)]

    @staticmethod
    def _softmax(actions: List[Action], scores: List[float], tau: float, rng: random.Random) -> Action:
        if not actions:
            return ("move", None)
        if len(actions) == 1:
            return actions[0]

        tau = max(float(tau), 1e-6)
        m = max(scores)
        exps = [math.exp((s - m) / tau) for s in scores]
        total = sum(exps)

        if total <= 0.0 or not math.isfinite(total):
            return rng.choice(actions)

        r = rng.random() * total
        acc = 0.0
        for a, e in zip(actions, exps):
            acc += e
            if acc >= r:
                return a
        return actions[-1]

    def _apply_end_of_turn_chip(self) -> "ShadowState":
        CHIP_BRN = 1.0 / 16.0  # burn chip per turn in Gen 9
        CHIP_PSN = 1.0 / 8.0   # regular poison chip (approx)
        new_my_hp = dict(self.my_hp)
        new_opp_hp = dict(self.opp_hp)

        new_my_toxic_counter = self.my_toxic_counter
        new_opp_toxic_counter = self.opp_toxic_counter

        # Active-only chip (good approximation for planning)
        if self.my_status.get(id(self.my_active)) == Status.BRN:
            new_my_hp[id(self.my_active)] = max(0.0, new_my_hp.get(id(self.my_active), 0.0) - CHIP_BRN)

        if self.opp_status.get(id(self.opp_active)) == Status.BRN:
            new_opp_hp[id(self.opp_active)] = max(0.0, new_opp_hp.get(id(self.opp_active), 0.0) - CHIP_BRN)

        # Poison / toxic (active-only chip approximation)
        my_st = self.my_status.get(id(self.my_active))
        if my_st == Status.PSN:
            new_my_hp[id(self.my_active)] = max(0.0, new_my_hp.get(id(self.my_active), 0.0) - CHIP_PSN)
        elif my_st == Status.TOX:
            # Increment toxic counter
            new_my_toxic_counter += 1
            # Toxic damage scales: 1/16, 2/16, 3/16, etc.
            toxic_damage = new_my_toxic_counter / 16.0
            new_my_hp[id(self.my_active)] = max(0.0, new_my_hp.get(id(self.my_active), 0.0) - toxic_damage)

        opp_st = self.opp_status.get(id(self.opp_active))
        if opp_st == Status.PSN:
            new_opp_hp[id(self.opp_active)] = max(0.0, new_opp_hp.get(id(self.opp_active), 0.0) - CHIP_PSN)
        elif opp_st == Status.TOX:
            # Increment toxic counter
            new_opp_toxic_counter += 1
            # Toxic damage scales: 1/16, 2/16, 3/16, etc.
            toxic_damage = new_opp_toxic_counter / 16.0
            new_opp_hp[id(self.opp_active)] = max(0.0, new_opp_hp.get(id(self.opp_active), 0.0) - toxic_damage)

        # Leftovers / Black Sludge end-of-turn recovery
        for active, hp_map in [
            (self.my_active, new_my_hp),
            (self.opp_active, new_opp_hp),
        ]:
            if hp_map.get(id(active), 0.0) <= 0.0:
                continue
            item = _get_item(active)
            if item == 'leftovers':
                hp_map[id(active)] = min(1.0, hp_map[id(active)] + 1.0 / 16.0)
            elif item == 'blacksludge':
                is_poison = PokemonType.POISON in getattr(active, 'types', [])
                if is_poison:
                    hp_map[id(active)] = min(1.0, hp_map[id(active)] + 1.0 / 16.0)
                else:
                    hp_map[id(active)] = max(0.0, hp_map[id(active)] - 1.0 / 16.0)

        # Sandstorm chip: 1/16 to non-Rock/Steel/Ground actives
        if Weather.SANDSTORM in self.shadow_weather:
            for active, hp_map in [(self.my_active, new_my_hp), (self.opp_active, new_opp_hp)]:
                if hp_map.get(id(active), 0.0) <= 0.0:
                    continue
                types = getattr(active, 'types', [])
                immune = any(t in (PokemonType.ROCK, PokemonType.STEEL, PokemonType.GROUND) for t in types)
                if not immune:
                    hp_map[id(active)] = max(0.0, hp_map[id(active)] - 1.0 / 16.0)

        # Grassy Terrain heal: 1/16 to grounded actives
        if Field.GRASSY_TERRAIN in self.shadow_fields:
            for active, hp_map in [(self.my_active, new_my_hp), (self.opp_active, new_opp_hp)]:
                if hp_map.get(id(active), 0.0) <= 0.0:
                    continue
                if _is_grounded(active):
                    hp_map[id(active)] = min(1.0, hp_map[id(active)] + 1.0 / 16.0)

        # Decrement weather counters
        new_weather = {}
        for w, t in self.shadow_weather.items():
            t2 = t + 1
            if t2 < 5:
                new_weather[w] = t2

        # Decrement field counters (terrain, trick room)
        new_fields = {}
        for f, t in self.shadow_fields.items():
            t2 = t + 1
            if t2 < 5:
                new_fields[f] = t2

        # Decrement timed side conditions (screens, tailwind)
        def _dec_sc(sc_dict: Dict[str, int]) -> Dict[str, int]:
            new = {}
            for k, v in sc_dict.items():
                if k in TIMED_SIDE_CONDITIONS:
                    if v > 1:
                        new[k] = v - 1
                    # else: expired, drop it
                else:
                    new[k] = v  # hazards don't expire
            return new

        new_my_sc = _dec_sc(self.my_side_conditions)
        new_opp_sc = _dec_sc(self.opp_side_conditions)

        return replace(self, my_hp=new_my_hp, opp_hp=new_opp_hp,
                      my_toxic_counter=new_my_toxic_counter, opp_toxic_counter=new_opp_toxic_counter,
                      shadow_weather=new_weather, shadow_fields=new_fields,
                      my_side_conditions=new_my_sc, opp_side_conditions=new_opp_sc)

    def _sample_hit(self, move: Any, rng: random.Random) -> bool:
        # Check for forced outcome first (for hybrid expansion)
        if hasattr(self, '_forced_hit'):
            return self._forced_hit
        
        if not self.model_miss:
            return True
        if move is None:
            return True

        acc = getattr(move, "accuracy", 1.0)

        if acc is None:
            acc = 1.0

        try:
            acc = float(acc)
        except Exception:
            acc = 1.0

        # handle % vs [0,1]
        if acc > 1.0:
            acc /= 100.0

        acc = max(0.0, min(1.0, acc))
        return rng.random() < acc


    def step(self, my_action: Action, *, rng: random.Random) -> "ShadowState":
        """
        One full turn transition using the provided RNG stream.
        This is what makes rollouts stochastic + reproducible.
        """
        if self.is_terminal():
            return self
        
        # Right before the speed comparison
        ms = self._effective_speed(self.my_active, "me")
        os = self._effective_speed(self.opp_active, "opp")

        opp_action = self.choose_opp_action(rng)
        order = self._order_for_turn(my_action, opp_action, rng)
        
        # print(f"\n=== ACTUAL TURN EXECUTION ===")
        # print(f"Before: {self.my_active.species} {self.my_active_hp():.0%} vs {self.opp_active.species} {self.opp_active_hp():.0%}")
        # print(f"My action: {my_action[0]} {getattr(my_action[1], 'id', getattr(my_action[1], 'species', '?'))}")
        # print(f"Opp action: {opp_action[0]} {getattr(opp_action[1], 'id', getattr(opp_action[1], 'species', '?'))}")
        # print(f"Order: {'+1 (we first)' if order == 1 else '-1 (opp first)'}")

        # Clear protect volatiles from previous turn
        s = self
        my_vol = s.my_volatiles.get(id(s.my_active), {})
        opp_vol = s.opp_volatiles.get(id(s.opp_active), {})
        my_had_protect = my_vol.get('protect', False)
        opp_had_protect = opp_vol.get('protect', False)
        if my_had_protect:
            new_vol = dict(s.my_volatiles)
            updated = {k: v for k, v in my_vol.items() if k != 'protect'}
            new_vol[id(s.my_active)] = updated
            s = replace(s, my_volatiles=new_vol)
        if opp_had_protect:
            new_vol = dict(s.opp_volatiles)
            updated = {k: v for k, v in opp_vol.items() if k != 'protect'}
            new_vol[id(s.opp_active)] = updated
            s = replace(s, opp_volatiles=new_vol)

        # Track whether each side uses Protect this turn (for counter reset)
        my_uses_protect = (my_action[0] == "move" and my_action[1] is not None
                          and str(getattr(my_action[1], 'id', '') or '').lower() in PROTECT_MOVES)
        opp_uses_protect = (opp_action[0] == "move" and opp_action[1] is not None
                           and str(getattr(opp_action[1], 'id', '') or '').lower() in PROTECT_MOVES)

        # Never let the second mover act if the first mover KO'd them (no damage/status after KO)
        HP_ALIVE = 1e-9

        if order == +1:
            s = s._apply_my_action(my_action, rng)
            if not s.is_terminal():
                opp_hp = s.opp_hp.get(id(s.opp_active), 0.0)
                if opp_hp > HP_ALIVE:
                    # Check flinch: our move may have flinched the opponent
                    if _check_flinch(my_action, rng):
                        s = s._log(f"FLINCH opp:{s.opp_active.species}")
                    else:
                        s = s._apply_opp_action(opp_action, rng)

        else:
            s = s._apply_opp_action(opp_action, rng)
            if not s.is_terminal():
                my_hp = s.my_hp.get(id(s.my_active), 0.0)
                if my_hp > HP_ALIVE:
                    # Check flinch: opponent's move may have flinched us
                    if _check_flinch(opp_action, rng):
                        s = s._log(f"FLINCH me:{s.my_active.species}")
                    else:
                        s = s._apply_my_action(my_action, rng)

        # Reset protect count if side didn't use Protect this turn
        if not my_uses_protect and s.my_protect_count > 0:
            s = replace(s, my_protect_count=0)
        if not opp_uses_protect and s.opp_protect_count > 0:
            s = replace(s, opp_protect_count=0)

        # End of full turn
        if not s.is_terminal():
            s = s._apply_end_of_turn_chip()

        # If an active fainted during the turn, force an automatic replacement.
        # This prevents illegal "fainted active keeps acting / being evaluated" states.
        if not s.is_terminal() and s.my_hp.get(id(s.my_active), 1.0) <= 0.0:
            if s.debug:
                print(f"[AUTO-SWITCH] My active {s.my_active.species} fainted (HP: {s.my_hp.get(id(s.my_active), 1.0):.1%})")

            # Capture the KO penalty BEFORE the auto-switch.
            # After switching, evaluate_state() sees my_active_hp > 0 and skips the -0.90
            # faint branch entirely — masking most of the penalty for getting KO'd.
            # Store it so evaluate_state() can return the correct value.
            opp_hp_now = s.opp_hp.get(id(s.opp_active), 0.0)
            faint_eval: Optional[float] = None
            if opp_hp_now > 0.0:
                my_sum = sum(max(0.0, min(1.0, v)) for v in s.my_hp.values())
                opp_sum = sum(max(0.0, min(1.0, v)) for v in s.opp_hp.values())
                lead_hint = math.tanh((my_sum - opp_sum) / 1.5)
                # Simple bench quality proxy: highest alive bench-mon HP fraction
                bench_hps = [v for pid, v in s.my_hp.items()
                             if pid != id(s.my_active) and v > 0.0]
                bench_qual = min(1.0, max(bench_hps)) if bench_hps else 0.0
                # Mirror the terminal eval in evaluate_state():
                #   base = -0.90 + 0.15 * lead_hint + 0.35 * bench_qual
                # Cap at 0.0: we never want a faint-turn to score positively.
                faint_eval = max(-1.0, min(0.0, -0.90 + 0.15 * lead_hint + 0.35 * bench_qual))

            target = s._best_my_switch()
            if target is not None:
                if s.debug:
                    print(f"[AUTO-SWITCH] → Switching to {target.species} (HP: {s.my_hp.get(id(target), 1.0):.1%})")
                s = s._apply_my_switch(target)
            else:
                if s.debug:
                    print(f"[AUTO-SWITCH] → No valid switch target!")

            if faint_eval is not None:
                s = replace(s, pre_autoswitch_eval=faint_eval)

        if not s.is_terminal() and s.opp_hp.get(id(s.opp_active), 1.0) <= 0.0:
            if s.debug:
                print(f"[AUTO-SWITCH] Opp active {s.opp_active.species} fainted (HP: {s.opp_hp.get(id(s.opp_active), 1.0):.1%})")
            target = s._best_opp_switch()
            if target is not None:
                if isinstance(target, tuple) and target[0] == "switch_unknown":
                    if s.debug:
                        print(f"[AUTO-SWITCH] → Switching to unknown slot {target[1]}")
                    s = s._apply_opp_switch_unknown(target[1], rng)
                else:
                    if s.debug:
                        print(f"[AUTO-SWITCH] → Switching to {target.species} (HP: {s.opp_hp.get(id(target), 1.0):.1%})")
                    s = s._apply_opp_switch(target)
            else:
                if s.debug:
                    print(f"[AUTO-SWITCH] → No valid switch target!")

        return replace(s, ply=s.ply + 1)

    def _apply_my_action(self, action: Action, rng: random.Random) -> "ShadowState":
        return self._apply_my_switch(action[1]) if action[0] == "switch" else self._apply_my_move(action[1], rng)

    def _apply_opp_action(self, action: Action, rng: random.Random) -> "ShadowState":
        if action[0] == "switch":
            return self._apply_opp_switch(action[1])
        if action[0] == "switch_unknown":
            return self._apply_opp_switch_unknown(action[1], rng)
        return self._apply_opp_move(action[1], rng)

    def _apply_field_effects(self, move: Any, side: str) -> "ShadowState":
        """Detect and apply weather/terrain/screen/tailwind/trick room setting from a move."""
        s = self
        if move is None:
            return s

        # Weather-setting moves
        move_weather = getattr(move, 'weather', None)
        if move_weather:
            s = replace(s, shadow_weather={move_weather: 0})

        # Terrain-setting moves (replace existing terrain, keep non-terrain fields)
        move_terrain = getattr(move, 'terrain', None)
        if move_terrain:
            new_fields = {k: v for k, v in s.shadow_fields.items()
                          if not getattr(k, 'is_terrain', False)}
            new_fields[move_terrain] = 0
            s = replace(s, shadow_fields=new_fields)

        # Trick Room (pseudo_weather) — toggles on/off
        pw = getattr(move, 'pseudo_weather', None)
        if pw and 'trickroom' in str(pw).lower().replace(' ', '').replace('_', ''):
            new_fields = dict(s.shadow_fields)
            if Field.TRICK_ROOM in new_fields:
                del new_fields[Field.TRICK_ROOM]
            else:
                new_fields[Field.TRICK_ROOM] = 0
            s = replace(s, shadow_fields=new_fields)

        # Side condition setting moves (Reflect, Light Screen, Aurora Veil, Tailwind)
        sc = getattr(move, 'side_condition', None)
        if sc:
            sc_name = getattr(sc, 'name', str(sc)).lower().replace('_', '')
            duration = 4 if sc_name == 'tailwind' else 5
            if side == "me":
                new_sc = dict(s.my_side_conditions)
                new_sc[sc_name] = duration
                s = replace(s, my_side_conditions=new_sc)
            else:
                new_sc = dict(s.opp_side_conditions)
                new_sc[sc_name] = duration
                s = replace(s, opp_side_conditions=new_sc)

        return s

    def _apply_my_switch(self, new_mon: Any) -> "ShadowState":
        if new_mon is None:
            return self
        # Only check shadow state HP, not the real Pokemon's fainted attribute
        if self.my_hp.get(id(new_mon), 1.0) <= 0.0:
            return self

        s = self._log(f"We switch to {getattr(new_mon, 'species', getattr(new_mon, 'name', 'unknown'))}")

        # Keep contexts aligned (best-effort)
        try:
            ctx_me = replace(s.ctx_me, me=new_mon, opp=s.opp_active)
        except Exception:
            ctx_me = s.ctx_me
            try:
                ctx_me.me = new_mon
                ctx_me.opp = s.opp_active
            except Exception:
                pass

        try:
            ctx_opp = replace(s.ctx_opp, opp=new_mon)
        except Exception:
            ctx_opp = s.ctx_opp
            try:
                ctx_opp.opp = new_mon
            except Exception:
                pass

        # Reset toxic counter, choice lock, volatiles, and protect count on switch
        new_vol = dict(s.my_volatiles)
        new_vol.pop(id(s.my_active), None)  # Old mon's volatiles cleared
        s = replace(s, my_active=new_mon, ctx_me=ctx_me, ctx_opp=ctx_opp,
                    my_toxic_counter=0, my_choice_lock=None,
                    my_volatiles=new_vol, my_protect_count=0)

        # Apply entry hazards from OUR side conditions
        h_dmg, h_status, h_spe, updated_sc = _apply_hazards_on_entry(
            new_mon, s.my_side_conditions, s.my_status.get(id(new_mon)))

        if h_dmg > 0.0:
            new_hp = dict(s.my_hp)
            new_hp[id(new_mon)] = max(0.0, new_hp.get(id(new_mon), 1.0) - h_dmg)
            s = replace(s, my_hp=new_hp)

        if h_status is not None and s.my_status.get(id(new_mon)) is None:
            new_st = dict(s.my_status)
            new_st[id(new_mon)] = h_status
            s = replace(s, my_status=new_st)

        if h_spe is not None:
            s = s._apply_boost_changes(h_spe, "me", new_mon)

        if updated_sc != s.my_side_conditions:
            s = replace(s, my_side_conditions=updated_sc)

        return s

    def _apply_my_move(self, move: Any, rng: random.Random) -> "ShadowState":
        s = self

        my_status = s.my_status.get(id(s.my_active))

        # Sleep check 
        if my_status == Status.SLP:
            my_vol = s.my_volatiles.get(id(s.my_active), {})
            sleep_turns = my_vol.get('sleep_turns', 0)
            if sleep_turns > 0:
                new_turns = sleep_turns - 1
                new_vol = dict(s.my_volatiles)
                if new_turns <= 0:
                    # Wake up — clear sleep status + volatile, act normally
                    updated = {k: v for k, v in my_vol.items() if k != 'sleep_turns'}
                    new_vol[id(s.my_active)] = updated
                    new_st = dict(s.my_status)
                    new_st[id(s.my_active)] = None
                    s = replace(s, my_status=new_st, my_volatiles=new_vol)
                    my_status = None  # update local var
                    s = s._log(f"WAKE me:{s.my_active.species}")
                else:
                    # Still asleep
                    new_vol[id(s.my_active)] = {**my_vol, 'sleep_turns': new_turns}
                    s = replace(s, my_volatiles=new_vol)
                    if getattr(move, 'sleep_usable', False):
                        s = s._log(f"SLEEPTALK me:{s.my_active.species}")
                        # Fall through to execute the move
                    else:
                        s = s._log(f"ASLEEP me:{s.my_active.species} ({new_turns} left)")
                        return s

        # Freeze check
        if my_status == Status.FRZ:
            move_type = getattr(move, 'type', None)
            is_fire = move_type is not None and move_type == PokemonType.FIRE
            if is_fire or rng.random() < 0.20:
                new_st = dict(s.my_status)
                new_st[id(s.my_active)] = None
                s = replace(s, my_status=new_st)
                my_status = None
                s = s._log(f"THAW me:{s.my_active.species}")
            else:
                s = s._log(f"FROZEN me:{s.my_active.species}")
                return s

        # Confusion check 
        my_vol = s.my_volatiles.get(id(s.my_active), {})
        conf_turns = my_vol.get('confusion_turns', 0)
        if conf_turns > 0:
            new_vol = dict(s.my_volatiles)
            new_turns = conf_turns - 1
            if new_turns <= 0:
                updated = {k: v for k, v in my_vol.items() if k != 'confusion_turns'}
            else:
                updated = {**my_vol, 'confusion_turns': new_turns}
            new_vol[id(s.my_active)] = updated
            s = replace(s, my_volatiles=new_vol)
            if rng.random() < 1.0 / 3.0:
                self_dmg = 0.05
                new_hp = dict(s.my_hp)
                new_hp[id(s.my_active)] = max(0.0, new_hp[id(s.my_active)] - self_dmg)
                s = replace(s, my_hp=new_hp)
                s = s._log(f"CONFUSED-SELFHIT me:{s.my_active.species}")
                return s

        # Paralysis check 
        if my_status == Status.PAR:
            if rng.random() < 0.25:
                s = s._log(f"FULL PARA me:{s.my_active.species}")
                return s

        # Psychic Terrain: priority moves fail against grounded targets
        if (move is not None
                and Field.PSYCHIC_TERRAIN in s.shadow_fields
                and move_priority(move) > 0
                and _is_grounded(s.opp_active)):
            s = s._log(f"PSYCHIC-TERRAIN-BLOCK me:{getattr(move,'id','move')}")
            return s

        # Protect check
        if move is not None:
            mid = str(getattr(move, 'id', '') or '').lower()
            if mid in PROTECT_MOVES:
                success_rate = 1.0 / (3 ** s.my_protect_count)
                if rng.random() < success_rate:
                    new_vol = dict(s.my_volatiles)
                    existing = new_vol.get(id(s.my_active), {})
                    new_vol[id(s.my_active)] = {**existing, 'protect': True}
                    s = replace(s, my_volatiles=new_vol, my_protect_count=s.my_protect_count + 1)
                    s = s._log(f"PROTECT me:{s.my_active.species}")
                else:
                    s = replace(s, my_protect_count=0)
                    s = s._log(f"PROTECT-FAIL me:{s.my_active.species}")
                return s

        # Check if opponent has Protect up — block damaging moves
        opp_vol = s.opp_volatiles.get(id(s.opp_active), {})
        if opp_vol.get('protect') and float(getattr(move, 'base_power', 0) or 0) > 0:
            s = s._log(f"BLOCKED-BY-PROTECT me:{getattr(move,'id','move')}")
            return s

        # Recovery moves (Recover, Roost, Slack Off, etc.) — heal 50% max HP
        move_heal = _safe_float(getattr(move, 'heal', 0))
        if move_heal > 0:
            new_hp = dict(s.my_hp)
            new_hp[id(s.my_active)] = min(1.0, new_hp[id(s.my_active)] + move_heal)
            s = replace(s, my_hp=new_hp)
            s = s._log(f"HEAL me:{s.my_active.species} +{move_heal:.0%}")
            return s

        hit = s._sample_hit(move, rng)

        if not hit:
            s = s._log(f"MISS me:{getattr(move,'id','move')}")
            # Crash damage on miss (High Jump Kick, Jump Kick)
            if (move is not None
                    and (getattr(move, 'entry', None) or {}).get('hasCrashDamage')):
                crash_hp = dict(s.my_hp)
                crash_hp[id(s.my_active)] = max(0.0, crash_hp[id(s.my_active)] - 0.5)
                s = replace(s, my_hp=crash_hp)
                s = s._log(f"CRASH me:{s.my_active.species} -50%")

        opp_hp_before = s.opp_active_hp()
        new_hp = dict(s.opp_hp)
        new_hp[id(s.opp_active)], did_crit = apply_expected_damage(
            s, move, s.my_active, s.opp_active, opp_hp_before, rng=rng, hit=hit
        )
        s = replace(s, opp_hp=new_hp)
        dmg_dealt = opp_hp_before - s.opp_active_hp()

        if did_crit:
            s = s._log(f"CRIT me:{getattr(move,'id','move')}")

        if hit:
            s = s._maybe_apply_status(move, "opp", s.opp_active, rng)

        if hit:
            s = s._maybe_apply_boosts_us(move, rng)

        if hit:
            s = s._maybe_apply_confusion(move, "opp", rng)

        # Drain healing (Giga Drain, Drain Punch, etc.)
        if hit and dmg_dealt > 0:
            drain_frac = _safe_float(getattr(move, 'drain', 0))
            if drain_frac > 0:
                heal_amt = dmg_dealt * drain_frac
                drain_hp = dict(s.my_hp)
                drain_hp[id(s.my_active)] = min(1.0, drain_hp[id(s.my_active)] + heal_amt)
                s = replace(s, my_hp=drain_hp)

        # Move recoil (Brave Bird, Flare Blitz, etc.)
        if hit and dmg_dealt > 0:
            recoil_frac = _safe_float(getattr(move, 'recoil', 0))
            if recoil_frac > 0:
                recoil_amt = dmg_dealt * recoil_frac
                recoil_hp = dict(s.my_hp)
                recoil_hp[id(s.my_active)] = max(0.0, recoil_hp[id(s.my_active)] - recoil_amt)
                s = replace(s, my_hp=recoil_hp)

        # Life Orb recoil: 1/10 max HP after dealing damage
        if hit and float(getattr(move, 'base_power', 0) or 0) > 0:
            if _get_item(s.my_active) == 'lifeorb':
                recoil_hp = dict(s.my_hp)
                recoil_hp[id(s.my_active)] = max(0.0, recoil_hp[id(s.my_active)] - 1.0 / 10.0)
                s = replace(s, my_hp=recoil_hp)

        # Hazard removal on hit
        if hit and move is not None:
            mid = str(getattr(move, 'id', '') or '').lower()
            if mid in HAZARD_REMOVAL_OWN_SIDE:
                # Rapid Spin / Mortal Spin / Tidy Up: clear our side
                cleared = {k: v for k, v in s.my_side_conditions.items() if k not in HAZARD_KEYS}
                if cleared != s.my_side_conditions:
                    s = replace(s, my_side_conditions=cleared)
            elif mid in HAZARD_REMOVAL_BOTH_SIDES:
                # Defog: clear both sides
                my_cleared = {k: v for k, v in s.my_side_conditions.items() if k not in HAZARD_KEYS}
                opp_cleared = {k: v for k, v in s.opp_side_conditions.items() if k not in HAZARD_KEYS}
                s = replace(s, my_side_conditions=my_cleared, opp_side_conditions=opp_cleared)

        # Field effects: weather/terrain/screens/tailwind/trick room setting
        s = s._apply_field_effects(move, "me")

        # Choice lock: lock into this move if holding a Choice item
        if move is not None and _get_item(s.my_active) in CHOICE_ITEMS:
            move_id = str(getattr(move, 'id', '') or '')
            if move_id:
                s = replace(s, my_choice_lock=move_id)

        if is_pivot_move(move) and hit:
            target = s._best_my_switch()
            if target is not None:
                s = s._apply_my_switch(target)

        # Clear forced outcomes after use (for hybrid expansion)
        if hasattr(s, '_forced_hit'):
            object.__delattr__(s, '_forced_hit')
        if hasattr(s, '_forced_crit'):
            object.__delattr__(s, '_forced_crit')

        return s


    def _apply_opp_switch_unknown(self, idx: int, rng: random.Random) -> "ShadowState":
        """Materialize a Pokémon from team_belief (without replacement) and switch to it."""
        team_belief = getattr(self, "team_belief", None)
        if team_belief is not None and team_belief.has_mass():
            new_mon, new_team_belief = sample_unseen_mon_from_team_belief(team_belief, self.gen, rng)
        else:
            known_ids = set()
            for p in self.opp_team:
                s = getattr(p, "species", None) or getattr(p, "base_species", None)
                if s:
                    known_ids.add(str(s).strip())
            new_mon = sample_unseen_mon(self.gen, known_ids, rng)
            new_team_belief = team_belief
        if new_mon is None:
            return self
        opp_slots = list(getattr(self, "opp_slots", None) or [None] * 6)
        while len(opp_slots) < 6:
            opp_slots.append(None)
        if idx < 0 or idx >= len(opp_slots):
            return self
        opp_slots[idx] = new_mon
        new_opp_team = list(self.opp_team) + [new_mon]
        new_opp_hp = dict(self.opp_hp)
        new_opp_hp[id(new_mon)] = 1.0
        new_opp_status = dict(self.opp_status)
        new_opp_status[id(new_mon)] = None
        new_opp_boosts = dict(self.opp_boosts)
        new_opp_boosts[id(new_mon)] = {
            "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0,
            "accuracy": 0, "evasion": 0,
        }
        s = replace(
            self,
            opp_slots=opp_slots,
            opp_team=new_opp_team,
            opp_hp=new_opp_hp,
            opp_status=new_opp_status,
            opp_boosts=new_opp_boosts,
            opp_active=new_mon,
            opp_active_idx=idx,
            team_belief=new_team_belief,
        )
        return s._apply_opp_switch(new_mon)

    def _apply_opp_switch(self, new_mon: Any) -> "ShadowState":
        if new_mon is None:
            return self
        # Only check shadow state HP, not the real Pokemon's fainted attribute
        # (The real Pokemon might be fainted, but in this shadow timeline it's alive)
        if self.opp_hp.get(id(new_mon), 1.0) <= 0.0:
            return self

        s = self._log(f"Opp switches to {getattr(new_mon, 'species', getattr(new_mon, 'name', 'unknown'))}")

        try:
            ctx_opp = replace(s.ctx_opp, me=new_mon, opp=s.my_active)
        except Exception:
            ctx_opp = s.ctx_opp
            try:
                ctx_opp.me = new_mon
                ctx_opp.opp = s.my_active
            except Exception:
                pass

        try:
            ctx_me = replace(s.ctx_me, opp=new_mon)
        except Exception:
            ctx_me = s.ctx_me
            try:
                ctx_me.opp = new_mon
            except Exception:
                pass

        # Reset toxic counter, choice lock, volatiles, and protect count on switch
        new_vol = dict(s.opp_volatiles)
        new_vol.pop(id(s.opp_active), None)  # Old mon's volatiles cleared
        new_idx = s.opp_active_idx
        for i, slot in enumerate(s.opp_slots):
            if slot is new_mon:
                new_idx = i
                break
        s = replace(s, opp_active=new_mon, opp_active_idx=new_idx, ctx_opp=ctx_opp, ctx_me=ctx_me,
                    opp_toxic_counter=0, opp_choice_lock=None,
                    opp_volatiles=new_vol, opp_protect_count=0)

        # Apply entry hazards from OPPONENT's side conditions
        h_dmg, h_status, h_spe, updated_sc = _apply_hazards_on_entry(
            new_mon, s.opp_side_conditions, s.opp_status.get(id(new_mon)))

        if h_dmg > 0.0:
            new_hp = dict(s.opp_hp)
            new_hp[id(new_mon)] = max(0.0, new_hp.get(id(new_mon), 1.0) - h_dmg)
            s = replace(s, opp_hp=new_hp)

        if h_status is not None and s.opp_status.get(id(new_mon)) is None:
            new_st = dict(s.opp_status)
            new_st[id(new_mon)] = h_status
            s = replace(s, opp_status=new_st)

        if h_spe is not None:
            s = s._apply_boost_changes(h_spe, "opp", new_mon)

        if updated_sc != s.opp_side_conditions:
            s = replace(s, opp_side_conditions=updated_sc)

        return s

    def _apply_opp_move(self, move: Any, rng: random.Random) -> "ShadowState":
        s = self

        opp_status = s.opp_status.get(id(s.opp_active))

        # Sleep check
        if opp_status == Status.SLP:
            opp_vol = s.opp_volatiles.get(id(s.opp_active), {})
            sleep_turns = opp_vol.get('sleep_turns', 0)
            if sleep_turns > 0:
                new_turns = sleep_turns - 1
                new_vol = dict(s.opp_volatiles)
                if new_turns <= 0:
                    updated = {k: v for k, v in opp_vol.items() if k != 'sleep_turns'}
                    new_vol[id(s.opp_active)] = updated
                    new_st = dict(s.opp_status)
                    new_st[id(s.opp_active)] = None
                    s = replace(s, opp_status=new_st, opp_volatiles=new_vol)
                    opp_status = None
                    s = s._log(f"WAKE opp:{s.opp_active.species}")
                else:
                    new_vol[id(s.opp_active)] = {**opp_vol, 'sleep_turns': new_turns}
                    s = replace(s, opp_volatiles=new_vol)
                    if getattr(move, 'sleep_usable', False):
                        s = s._log(f"SLEEPTALK opp:{s.opp_active.species}")
                    else:
                        s = s._log(f"ASLEEP opp:{s.opp_active.species} ({new_turns} left)")
                        return s

        # Freeze check 
        if opp_status == Status.FRZ:
            move_type = getattr(move, 'type', None)
            is_fire = move_type is not None and move_type == PokemonType.FIRE
            if is_fire or rng.random() < 0.20:
                new_st = dict(s.opp_status)
                new_st[id(s.opp_active)] = None
                s = replace(s, opp_status=new_st)
                opp_status = None
                s = s._log(f"THAW opp:{s.opp_active.species}")
            else:
                s = s._log(f"FROZEN opp:{s.opp_active.species}")
                return s

        # Confusion check 
        opp_vol = s.opp_volatiles.get(id(s.opp_active), {})
        conf_turns = opp_vol.get('confusion_turns', 0)
        if conf_turns > 0:
            new_vol = dict(s.opp_volatiles)
            new_turns = conf_turns - 1
            if new_turns <= 0:
                updated = {k: v for k, v in opp_vol.items() if k != 'confusion_turns'}
            else:
                updated = {**opp_vol, 'confusion_turns': new_turns}
            new_vol[id(s.opp_active)] = updated
            s = replace(s, opp_volatiles=new_vol)
            if rng.random() < 1.0 / 3.0:
                self_dmg = 0.05
                new_hp = dict(s.opp_hp)
                new_hp[id(s.opp_active)] = max(0.0, new_hp[id(s.opp_active)] - self_dmg)
                s = replace(s, opp_hp=new_hp)
                s = s._log(f"CONFUSED-SELFHIT opp:{s.opp_active.species}")
                return s

        # Paralysis check
        if opp_status == Status.PAR:
            if rng.random() < 0.25:
                s = s._log(f"FULL PARA opp:{s.opp_active.species}")
                return s

        # Psychic Terrain: priority moves fail against grounded targets
        if (move is not None
                and Field.PSYCHIC_TERRAIN in s.shadow_fields
                and move_priority(move) > 0
                and _is_grounded(s.my_active)):
            s = s._log(f"PSYCHIC-TERRAIN-BLOCK opp:{getattr(move,'id','move')}")
            return s

        # Protect check
        if move is not None:
            mid = str(getattr(move, 'id', '') or '').lower()
            if mid in PROTECT_MOVES:
                success_rate = 1.0 / (3 ** s.opp_protect_count)
                if rng.random() < success_rate:
                    new_vol = dict(s.opp_volatiles)
                    existing = new_vol.get(id(s.opp_active), {})
                    new_vol[id(s.opp_active)] = {**existing, 'protect': True}
                    s = replace(s, opp_volatiles=new_vol, opp_protect_count=s.opp_protect_count + 1)
                    s = s._log(f"PROTECT opp:{s.opp_active.species}")
                else:
                    s = replace(s, opp_protect_count=0)
                    s = s._log(f"PROTECT-FAIL opp:{s.opp_active.species}")
                return s

        # Check if we have Protect up — block opponent's damaging moves
        my_vol = s.my_volatiles.get(id(s.my_active), {})
        if my_vol.get('protect') and float(getattr(move, 'base_power', 0) or 0) > 0:
            s = s._log(f"BLOCKED-BY-PROTECT opp:{getattr(move,'id','move')}")
            return s

        # Recovery moves (Recover, Roost, Slack Off, etc.)
        move_heal = _safe_float(getattr(move, 'heal', 0))
        if move_heal > 0:
            new_hp = dict(s.opp_hp)
            new_hp[id(s.opp_active)] = min(1.0, new_hp[id(s.opp_active)] + move_heal)
            s = replace(s, opp_hp=new_hp)
            s = s._log(f"HEAL opp:{s.opp_active.species} +{move_heal:.0%}")
            return s

        hit = s._sample_hit(move, rng)

        if not hit:
            s = s._log(f"MISS opp:{getattr(move,'id','move')}")
            # Crash damage on miss (High Jump Kick, Jump Kick)
            if (move is not None
                    and (getattr(move, 'entry', None) or {}).get('hasCrashDamage')):
                crash_hp = dict(s.opp_hp)
                crash_hp[id(s.opp_active)] = max(0.0, crash_hp[id(s.opp_active)] - 0.5)
                s = replace(s, opp_hp=crash_hp)
                s = s._log(f"CRASH opp:{s.opp_active.species} -50%")

        my_hp_before = s.my_active_hp()
        new_hp = dict(s.my_hp)
        new_hp[id(s.my_active)], did_crit = apply_expected_damage(
            s, move, s.opp_active, s.my_active, my_hp_before, rng=rng, hit=hit
        )
        s = replace(s, my_hp=new_hp)
        dmg_dealt = my_hp_before - s.my_active_hp()

        if did_crit:
            s = s._log(f"CRIT opp:{getattr(move,'id','move')}")

        if hit:
            s = s._maybe_apply_status(move, "me", s.my_active, rng)

        if hit:
            s = s._maybe_apply_boosts_opp(move, rng)

        if hit:
            s = s._maybe_apply_confusion(move, "me", rng)

        # Drain healing (Giga Drain, Drain Punch, etc.)
        if hit and dmg_dealt > 0:
            drain_frac = _safe_float(getattr(move, 'drain', 0))
            if drain_frac > 0:
                heal_amt = dmg_dealt * drain_frac
                drain_hp = dict(s.opp_hp)
                drain_hp[id(s.opp_active)] = min(1.0, drain_hp[id(s.opp_active)] + heal_amt)
                s = replace(s, opp_hp=drain_hp)

        # Move recoil (Brave Bird, Flare Blitz, etc.)
        if hit and dmg_dealt > 0:
            recoil_frac = _safe_float(getattr(move, 'recoil', 0))
            if recoil_frac > 0:
                recoil_amt = dmg_dealt * recoil_frac
                recoil_hp = dict(s.opp_hp)
                recoil_hp[id(s.opp_active)] = max(0.0, recoil_hp[id(s.opp_active)] - recoil_amt)
                s = replace(s, opp_hp=recoil_hp)

        # Life Orb recoil: 1/10 max HP after dealing damage
        if hit and float(getattr(move, 'base_power', 0) or 0) > 0:
            if _get_item(s.opp_active) == 'lifeorb':
                recoil_hp = dict(s.opp_hp)
                recoil_hp[id(s.opp_active)] = max(0.0, recoil_hp[id(s.opp_active)] - 1.0 / 10.0)
                s = replace(s, opp_hp=recoil_hp)

        # Hazard removal on hit
        if hit and move is not None:
            mid = str(getattr(move, 'id', '') or '').lower()
            if mid in HAZARD_REMOVAL_OWN_SIDE:
                # Rapid Spin / Mortal Spin / Tidy Up: clear opponent's side
                cleared = {k: v for k, v in s.opp_side_conditions.items() if k not in HAZARD_KEYS}
                if cleared != s.opp_side_conditions:
                    s = replace(s, opp_side_conditions=cleared)
            elif mid in HAZARD_REMOVAL_BOTH_SIDES:
                # Defog: clear both sides
                my_cleared = {k: v for k, v in s.my_side_conditions.items() if k not in HAZARD_KEYS}
                opp_cleared = {k: v for k, v in s.opp_side_conditions.items() if k not in HAZARD_KEYS}
                s = replace(s, my_side_conditions=my_cleared, opp_side_conditions=opp_cleared)

        # Field effects: weather/terrain/screens/tailwind/trick room setting
        s = s._apply_field_effects(move, "opp")

        # Choice lock: lock into this move if holding a Choice item
        if move is not None and _get_item(s.opp_active) in CHOICE_ITEMS:
            move_id = str(getattr(move, 'id', '') or '')
            if move_id:
                s = replace(s, opp_choice_lock=move_id)

        if is_pivot_move(move) and hit:
            target = s._best_opp_switch()
            if target is not None:
                if isinstance(target, tuple) and target[0] == "switch_unknown":
                    s = s._apply_opp_switch_unknown(target[1], rng)
                else:
                    s = s._apply_opp_switch(target)

        return s


    def _maybe_apply_status(self, move: Any, side: str, defender: Any, rng: random.Random) -> "ShadowState":
        info = status_infliction(move)
        if not info:
            return self

        st, prob = info
        if prob < self.status_threshold:
            return self

        # Terrain-based status immunity
        if _is_grounded(defender):
            # Electric Terrain: grounded Pokemon can't fall asleep
            if st == Status.SLP and Field.ELECTRIC_TERRAIN in self.shadow_fields:
                return self
            # Misty Terrain: grounded Pokemon can't be statused
            if Field.MISTY_TERRAIN in self.shadow_fields:
                return self

        # stochastic proc
        if prob < 1.0 and rng.random() >= float(prob):
            return self

        if side == "me":
            if self.my_status.get(id(defender)) is not None:
                return self
            mp = dict(self.my_status)
            mp[id(defender)] = st
            s = replace(self, my_status=mp)
            # Set sleep counter when SLP is inflicted (Gen 9: 1-3 turns)
            if st == Status.SLP:
                vol = dict(s.my_volatiles)
                existing = vol.get(id(defender), {})
                vol[id(defender)] = {**existing, 'sleep_turns': rng.randint(1, 3)}
                s = replace(s, my_volatiles=vol)
            return s
        else:
            if self.opp_status.get(id(defender)) is not None:
                return self
            mp = dict(self.opp_status)
            mp[id(defender)] = st
            s = replace(self, opp_status=mp)
            # Set sleep counter when SLP is inflicted (Gen 9: 1-3 turns)
            if st == Status.SLP:
                vol = dict(s.opp_volatiles)
                existing = vol.get(id(defender), {})
                vol[id(defender)] = {**existing, 'sleep_turns': rng.randint(1, 3)}
                s = replace(s, opp_volatiles=vol)
            return s
        
    def _maybe_apply_confusion(self, move: Any, side: str, rng: random.Random) -> "ShadowState":
        """Check if move inflicts confusion and apply it."""
        chance = 0.0

        # Check volatile_status (status moves like Confuse Ray)
        vol_status = getattr(move, 'volatile_status', None)
        if vol_status and 'confusion' in str(vol_status).lower():
            chance = 1.0
        else:
            # Check secondary effects (Hurricane, Dynamic Punch, etc.)
            secondary = getattr(move, 'secondary', None)
            if secondary and isinstance(secondary, list):
                for sec in secondary:
                    if isinstance(sec, dict) and sec.get('volatileStatus') == 'confusion':
                        chance = sec.get('chance', 100) / 100.0
                        break

        if chance == 0.0 or chance < self.status_threshold:
            return self
        if chance < 1.0 and rng.random() >= chance:
            return self

        # Apply confusion (2-5 turns in Gen 9)
        target_id = id(self.my_active) if side == "me" else id(self.opp_active)
        vol_key = "my_volatiles" if side == "me" else "opp_volatiles"
        vol_map = dict(getattr(self, vol_key))
        existing = vol_map.get(target_id, {})
        if 'confusion_turns' not in existing:
            vol_map[target_id] = {**existing, 'confusion_turns': rng.randint(2, 5)}
            species = self.my_active.species if side == "me" else self.opp_active.species
            s = replace(self, **{vol_key: vol_map})
            return s._log(f"CONFUSED {side}:{species}")
        return self

    def _apply_boost_changes(self, boost_changes: Dict[str, int], side: str, pokemon: Any) -> "ShadowState":
        """
        Apply boost changes to a Pokemon, clamping to [-6, +6].
        
        Args:
            boost_changes: Dict like {'atk': 2, 'spe': 1}
            side: 'me' or 'opp'
            pokemon: The Pokemon receiving the boosts
        
        Returns:
            New ShadowState with updated boosts
        """
        if side == "me":
            boosts_map = dict(self.my_boosts)
            current = boosts_map.get(id(pokemon), {
                'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0,
                'accuracy': 0, 'evasion': 0
            }).copy()
        else:
            boosts_map = dict(self.opp_boosts)
            current = boosts_map.get(id(pokemon), {
                'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0,
                'accuracy': 0, 'evasion': 0
            }).copy()
        
        # Apply changes, clamp to [-6, +6]
        for stat, change in boost_changes.items():
            if stat in current:
                new_val = current[stat] + change
                current[stat] = max(-6, min(6, new_val))
        
        boosts_map[id(pokemon)] = current
        
        if side == "me":
            return replace(self, my_boosts=boosts_map)
        else:
            return replace(self, opp_boosts=boosts_map)


    def _maybe_apply_boosts_us(self, move: Any, rng: random.Random) -> "ShadowState":
        """
        Apply stat boosts when we use a move.
        
        Self-boosts go to us, target-boosts go to opponent.
        """
        boost_data = get_move_boosts(move)
        if not boost_data:
            return self
        
        self_boosts, target_boosts, chance = boost_data
        
        # Roll for chance
        if chance < 1.0 and rng.random() >= float(chance):
            return self  # Didn't proc
        
        s = self
        
        # Apply self-boosts (to us)
        if self_boosts:
            s = s._apply_boost_changes(self_boosts, "me", s.my_active)
            if s.debug:
                boost_str = ", ".join(f"{k}:{v:+d}" for k, v in self_boosts.items())
                s = s._log(f"BOOST me:{s.my_active.species} {boost_str}")
        
        # Apply target-boosts (to opponent)
        if target_boosts:
            s = s._apply_boost_changes(target_boosts, "opp", s.opp_active)
            if s.debug:
                boost_str = ", ".join(f"{k}:{v:+d}" for k, v in target_boosts.items())
                s = s._log(f"BOOST opp:{s.opp_active.species} {boost_str}")
        
        return s


    def _maybe_apply_boosts_opp(self, move: Any, rng: random.Random) -> "ShadowState":
        """
        Apply stat boosts when opponent uses a move.
        
        Self-boosts go to opponent, target-boosts go to us.
        """
        boost_data = get_move_boosts(move)
        if not boost_data:
            return self
        
        self_boosts, target_boosts, chance = boost_data
        
        # Roll for chance
        if chance < 1.0 and rng.random() >= float(chance):
            return self
        
        s = self
        
        # Apply self-boosts (to opponent)
        if self_boosts:
            s = s._apply_boost_changes(self_boosts, "opp", s.opp_active)
            if s.debug:
                boost_str = ", ".join(f"{k}:{v:+d}" for k, v in self_boosts.items())
                s = s._log(f"BOOST opp:{s.opp_active.species} {boost_str}")
        
        # Apply target-boosts (to us)
        if target_boosts:
            s = s._apply_boost_changes(target_boosts, "me", s.my_active)
            if s.debug:
                boost_str = ", ".join(f"{k}:{v:+d}" for k, v in target_boosts.items())
                s = s._log(f"BOOST me:{s.my_active.species} {boost_str}")
        
        return s

    def _best_my_switch(self) -> Optional[Any]:
        best, val = None, -1e18
        with self._patched_status():
            for p in self.my_team:
                if p is self.my_active or is_fainted(p) or self.my_hp.get(id(p), 1.0) <= 0.0:
                    continue
                sc = float(self.score_switch_fn(p, self.battle, self.ctx_me))
                if sc > val:
                    best, val = p, sc
        return best

    def _best_opp_switch(self) -> Optional[Any]:
        """Returns best known switch target, or ('switch_unknown', idx) if only unknowns left."""
        best, val = None, -1e18
        with self._patched_status():
            for p in self.opp_team:
                if p is self.opp_active or is_fainted(p) or self.opp_hp.get(id(p), 1.0) <= 0.0:
                    continue
                sc = float(self.score_switch_fn(p, self.battle, self.ctx_opp))
                if sc > val:
                    best, val = p, sc
        if best is not None:
            return best
        tb = getattr(self, "team_belief", None)
        if tb is None or not tb.has_mass():
            return None
        opp_slots = getattr(self, "opp_slots", None) or [None] * 6
        for idx in range(min(6, len(opp_slots))):
            if opp_slots[idx] is None:
                return ("switch_unknown", idx)
        return None
    
    def _log(self, msg: str) -> "ShadowState":
        """Append a debug event.

        IMPORTANT: we prefix with the current ply so downstream tooling can
        reliably attribute events to a single simulated turn.
        """
        if not getattr(self, "debug", False):
            return self
        return replace(self, events=self.events + (f"[P{self.ply}] {msg}",))