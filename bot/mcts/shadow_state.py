from __future__ import annotations

from dataclasses import dataclass, replace, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from contextlib import contextmanager
import math
import random

from poke_env.battle import Status, MoveCategory

from bot.scoring.helpers import hp_frac, is_fainted

Action = Tuple[str, Any]  # ("move", Move) or ("switch", Pokemon)

EvalContext = Any
ScoreMoveFn = Callable[[Any, Any, EvalContext], float]
ScoreSwitchFn = Callable[[Any, Any, EvalContext], float]
DamageFn = Callable[[Any, Any, Any, Any], float]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def is_pivot_move(move: Any) -> bool:
    mid = str(getattr(move, "id", "") or getattr(move, "name", "")).lower().replace(" ", "")
    return mid in {"voltswitch", "uturn", "flipturn", "partingshot"}


def move_priority(move: Any) -> int:
    try:
        return int(getattr(move, "priority", 0) or 0)
    except Exception:
        return 0


def base_speed(p: Any, default: int = 80) -> float:
    try:
        return float((p.base_stats or {}).get("spe", default))
    except Exception:
        return float(default)


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
    
    Examples:
    - Swords Dance: ({'atk': 2}, None, 1.0)
    - Dragon Dance: ({'atk': 1, 'spe': 1}, None, 1.0)
    - Draco Meteor: ({'spa': -2}, None, 1.0)
    - Charm: (None, {'atk': -2}, 1.0)
    - Crunch: (None, {'def': -1}, 0.2)  # 20% chance
    """
    # Check for guaranteed self-boost (e.g., Swords Dance, Dragon Dance)
    boosts = getattr(move, 'boosts', None)
    if boosts:
        return (boosts, None, 1.0)
    
    # Check move.self for self-inflicted changes (e.g., Draco Meteor, Superpower)
    # NOTE: for the future I'd probably want to check abilities here and see if Contrary or any abilities that influence stat drops are involved
    self_data = getattr(move, 'self', None)
    if self_data and isinstance(self_data, dict):
        self_boosts = self_data.get('boosts', None)
        if self_boosts:
            return (self_boosts, None, 1.0)
    
    # Check secondary effects for chance-based boosts
    secondary = getattr(move, 'secondary', None)
    if secondary:
        if isinstance(secondary, list):
            for sec in secondary:
                if isinstance(sec, dict):
                    sec_boosts = sec.get('boosts', None)
                    if sec_boosts:
                        chance = sec.get('chance', 100) / 100.0
                        
                        # Check if it's a self-boost or target-boost
                        # If 'self' key exists in secondary, it's a self-boost
                        if sec.get('self'):
                            return (sec_boosts, None, chance)
                        else:
                            # Most secondary boosts affect the target
                            return (None, sec_boosts, chance)
        
        elif isinstance(secondary, dict):
            # Sometimes secondary is a dict, not a list
            sec_boosts = secondary.get('boosts', None)
            if sec_boosts:
                chance = secondary.get('chance', 100) / 100.0
                if secondary.get('self'):
                    return (sec_boosts, None, chance)
                else:
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
        with state._patched_status(), state._patched_boosts():
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

    ply: int = 0

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
        debug: bool = False
    ) -> "ShadowState":
        me = ctx_me.me
        opp = ctx_me.opp

        my_team = [p for p in getattr(battle, "team", {}).values() if p]
        opp_team = [p for p in getattr(battle, "opponent_team", {}).values() if p]

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
            my_hp=my_hp,
            opp_hp=opp_hp,
            my_status=my_status,
            opp_status=opp_status,
            my_boosts=my_boosts,
            opp_boosts=opp_boosts, 
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
        )

    @contextmanager
    def _patched_status(self):
        my_p, opp_p = self.my_active, self.opp_active
        old_my = getattr(my_p, "status", None)
        old_opp = getattr(opp_p, "status", None)

        try:
            my_p.status = self.my_status.get(id(my_p), old_my)
            opp_p.status = self.opp_status.get(id(opp_p), old_opp)
            yield
        finally:
            my_p.status = old_my
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

    def _effective_speed(self, p: Any, side: str) -> float:
        s = base_speed(p)
        st = (self.my_status if side == "me" else self.opp_status).get(id(p))
        if st == Status.PAR:
            s *= 0.5
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

        if ms != os:
            return +1 if ms > os else -1

        # speed tie
        return +1 if rng.random() < 0.5 else -1

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
            return actions or [("move", None)]

        for m in (getattr(self.opp_active, "moves", None) or {}).values():
            actions.append(("move", m))

        for p in self.opp_team:
            if p is self.opp_active:
                continue
            if is_fainted(p) or self.opp_hp.get(id(p), 1.0) <= 0.0:
                continue
            actions.append(("switch", p))

        return actions or [("move", None)]


        for m in (getattr(self.opp_active, "moves", None) or {}).values():
            actions.append(("move", m))

        for p in self.opp_team:
            if p is self.opp_active:
                continue
            if is_fainted(p) or self.opp_hp.get(id(p), 1.0) <= 0.0:
                continue
            actions.append(("switch", p))

        return actions or [("move", None)]

    def choose_opp_action(self, rng: random.Random) -> Action:
        actions = self.legal_actions_opp()
        scores: List[float] = []

        with self._patched_status(), self._patched_boosts():
            for k, o in actions:
                if k == "move":
                    scores.append(float(self.score_move_fn(o, self.battle, self.ctx_opp)))
                else:
                    scores.append(float(self.score_switch_fn(o, self.battle, self.ctx_opp)))

        return self._softmax(actions, scores, self.opp_tau, rng)

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
        CHIP_TOX = 1.0 / 6.0   # toxic chip (approx; we don't model counter here)
        new_my_hp = dict(self.my_hp)
        new_opp_hp = dict(self.opp_hp)

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
            new_my_hp[id(self.my_active)] = max(0.0, new_my_hp.get(id(self.my_active), 0.0) - CHIP_TOX)

        opp_st = self.opp_status.get(id(self.opp_active))
        if opp_st == Status.PSN:
            new_opp_hp[id(self.opp_active)] = max(0.0, new_opp_hp.get(id(self.opp_active), 0.0) - CHIP_PSN)
        elif opp_st == Status.TOX:
            new_opp_hp[id(self.opp_active)] = max(0.0, new_opp_hp.get(id(self.opp_active), 0.0) - CHIP_TOX)

        return replace(self, my_hp=new_my_hp, opp_hp=new_opp_hp)

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

        opp_action = self.choose_opp_action(rng)
        order = self._order_for_turn(my_action, opp_action, rng)
        
        print(f"\n=== ACTUAL TURN EXECUTION ===")
        print(f"Before: {self.my_active.species} {self.my_active_hp():.0%} vs {self.opp_active.species} {self.opp_active_hp():.0%}")
        print(f"My action: {my_action[0]} {getattr(my_action[1], 'id', getattr(my_action[1], 'species', '?'))}")
        print(f"Opp action: {opp_action[0]} {getattr(opp_action[1], 'id', getattr(opp_action[1], 'species', '?'))}")
        print(f"Order: {'+1 (we first)' if order == 1 else '-1 (opp first)'}")

        s = self
        if order == +1:
            print(">>> Applying our action...")
            s = s._apply_my_action(my_action, rng)
            # Don't allow fainted Pokemon to use moves (but allow switches)
            print(f"After our action: {s.my_active.species} {s.my_active_hp():.0%} vs {s.opp_active.species} {s.opp_active_hp():.0%}")
        
            if not s.is_terminal():
                if opp_action[0] == "switch" or s.opp_hp.get(id(s.opp_active), 0.0) > 0.0:
                    print(">>> Applying opponent's action...")
                    s = s._apply_opp_action(opp_action, rng)
                    print(f"After opp action: {s.my_active.species} {s.my_active_hp():.0%} vs {s.opp_active.species} {s.opp_active_hp():.0%}")
                else:
                    print(">>> Opponent's action SKIPPED (fainted)")

        else:
            s = s._apply_opp_action(opp_action, rng)
            # Don't allow fainted Pokemon to use moves (but allow switches)
            if not s.is_terminal():
                if my_action[0] == "switch" or s.my_hp.get(id(s.my_active), 0.0) > 0.0:
                    s = s._apply_my_action(my_action, rng)

        # End of full turn
        if not s.is_terminal():
            s = s._apply_end_of_turn_chip()

        # If an active fainted during the turn, force an automatic replacement.
        # This prevents illegal "fainted active keeps acting / being evaluated" states.
        if not s.is_terminal() and s.my_hp.get(id(s.my_active), 1.0) <= 0.0:
            target = s._best_my_switch()
            if target is not None:
                s = s._apply_my_switch(target)

        if not s.is_terminal() and s.opp_hp.get(id(s.opp_active), 1.0) <= 0.0:
            print(f"[AUTO-SWITCH] Opp active {s.opp_active.species} at {s.opp_hp.get(id(s.opp_active), 1.0):.1%}")
            target = s._best_opp_switch()
            print(f"[AUTO-SWITCH] Switching to {target.species if target else None}")
            if target is not None:
                s = s._apply_opp_switch(target)

        return replace(s, ply=s.ply + 1)

    def _apply_my_action(self, action: Action, rng: random.Random) -> "ShadowState":
        return self._apply_my_switch(action[1]) if action[0] == "switch" else self._apply_my_move(action[1], rng)

    def _apply_opp_action(self, action: Action, rng: random.Random) -> "ShadowState":
        return self._apply_opp_switch(action[1]) if action[0] == "switch" else self._apply_opp_move(action[1], rng)


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

        return replace(s, my_active=new_mon, ctx_me=ctx_me, ctx_opp=ctx_opp)

    def _apply_my_move(self, move: Any, rng: random.Random) -> "ShadowState":
        s = self
        hit = s._sample_hit(move, rng)

        if not hit:
            s = s._log(f"MISS me:{getattr(move,'id','move')}")

        new_hp = dict(s.opp_hp)
        new_hp[id(s.opp_active)], did_crit = apply_expected_damage(
            s, move, s.my_active, s.opp_active, s.opp_active_hp(), rng=rng, hit=hit
        )
        s = replace(s, opp_hp=new_hp)

        if did_crit:
            s = s._log(f"CRIT me:{getattr(move,'id','move')}")

        if hit:
            s = s._maybe_apply_status(move, "opp", s.opp_active, rng)

        if hit:
            s = s._maybe_apply_boosts_us(move, rng)

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

        return replace(s, opp_active=new_mon, ctx_opp=ctx_opp, ctx_me=ctx_me)

    def _apply_opp_move(self, move: Any, rng: random.Random) -> "ShadowState":
        print(f"[_apply_opp_move] START: {self.opp_active.species} using {move.id}")
        print(f"[_apply_opp_move] Is pivot? {is_pivot_move(move)}")
        s = self
        hit = s._sample_hit(move, rng)

        if not hit:
            s = s._log(f"MISS opp:{getattr(move,'id','move')}")

        new_hp = dict(s.my_hp)
        new_hp[id(s.my_active)], did_crit = apply_expected_damage(
            s, move, s.opp_active, s.my_active, s.my_active_hp(), rng=rng, hit=hit
        )
        s = replace(s, my_hp=new_hp)

        if did_crit:
            s = s._log(f"CRIT opp:{getattr(move,'id','move')}")

        if hit:
            s = s._maybe_apply_status(move, "me", s.my_active, rng)

        if hit:
            s = s._maybe_apply_boosts_opp(move, rng)

        if is_pivot_move(move) and hit:
            print(f"[_apply_opp_move] PIVOT! Switching opponent...")
            target = s._best_opp_switch()
            if target is not None:
                s = s._apply_opp_switch(target)
                print(f"[_apply_opp_move] Switched to {s.opp_active.species}")
        
        print(f"[_apply_opp_move] END: {s.opp_active.species}")
        return s


    def _maybe_apply_status(self, move: Any, side: str, defender: Any, rng: random.Random) -> "ShadowState":
        info = status_infliction(move)
        if not info:
            return self

        st, prob = info
        if prob < self.status_threshold:
            return self

        # stochastic proc
        if prob < 1.0 and rng.random() >= float(prob):
            return self

        if side == "me":
            if self.my_status.get(id(defender)) is not None:
                return self
            mp = dict(self.my_status)
            mp[id(defender)] = st
            return replace(self, my_status=mp)
        else:
            if self.opp_status.get(id(defender)) is not None:
                return self
            mp = dict(self.opp_status)
            mp[id(defender)] = st
            return replace(self, opp_status=mp)
        
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
        best, val = None, -1e18
        with self._patched_status():
            for p in self.opp_team:
                if p is self.opp_active or is_fainted(p) or self.opp_hp.get(id(p), 1.0) <= 0.0:
                    continue
                sc = float(self.score_switch_fn(p, self.battle, self.ctx_opp))
                if sc > val:
                    best, val = p, sc
        return best
    
    def _log(self, msg: str) -> "ShadowState":
        """Append a debug event.

        IMPORTANT: we prefix with the current ply so downstream tooling can
        reliably attribute events to a single simulated turn.
        """
        if not getattr(self, "debug", False):
            return self
        return replace(self, events=self.events + (f"[P{self.ply}] {msg}",))