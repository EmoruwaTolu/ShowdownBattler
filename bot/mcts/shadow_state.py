from __future__ import annotations

from dataclasses import dataclass, replace
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
    This is used by the forward model to apply *stochastic* procs.
    """
    mid = str(getattr(move, "id", "") or getattr(move, "name", "")).lower().replace(" ", "")

    # Guaranteed
    if mid == "willowisp":
        return (Status.BRN, 1.0)
    if mid in {"thunderwave", "glare", "nuzzle"}:
        return (Status.PAR, 1.0)

    # Common 30% procs
    burn_30 = {"lavaplume", "scald"}
    para_30 = {"bodyslam", "dragonbreath", "forcepalm", "discharge"}

    if mid in burn_30:
        return (Status.BRN, 0.30)
    if mid in para_30:
        return (Status.PAR, 0.30)

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

    try:
        with state._patched_status():
            dmg_frac = float(state.dmg_fn(move, attacker, defender, state.battle))
    except Exception:
        dmg_frac = 0.25
    
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

    if did_crit:
        state = state._log(f"CRIT {getattr(attacker,'species','att')}->{getattr(defender,'species','def')} via {getattr(move,'id','move')}")

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
        actions: List[Action] = []

        if self.ply == 0 and getattr(self.battle, "available_moves", None):
            for m in self.battle.available_moves:
                actions.append(("move", m))
        else:
            for m in (getattr(self.my_active, "moves", None) or {}).values():
                actions.append(("move", m))

        for p in self.my_team:
            if p is self.my_active:
                continue
            if is_fainted(p) or self.my_hp.get(id(p), 1.0) <= 0.0:
                continue
            actions.append(("switch", p))

        return actions or [("move", None)]

    def legal_actions_opp(self) -> List[Action]:
        actions: List[Action] = []

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

        with self._patched_status():
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
        CHIP = 1.0 / 16.0  # burn chip per turn in Gen 9
        new_my_hp = dict(self.my_hp)
        new_opp_hp = dict(self.opp_hp)

        # Active-only chip (good approximation for planning)
        if self.my_status.get(id(self.my_active)) == Status.BRN:
            new_my_hp[id(self.my_active)] = max(0.0, new_my_hp.get(id(self.my_active), 0.0) - CHIP)

        if self.opp_status.get(id(self.opp_active)) == Status.BRN:
            new_opp_hp[id(self.opp_active)] = max(0.0, new_opp_hp.get(id(self.opp_active), 0.0) - CHIP)

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

        s = self
        if order == +1:
            s = s._apply_my_action(my_action, rng)
            if not s.is_terminal():
                s = s._apply_opp_action(opp_action, rng)
        else:
            s = s._apply_opp_action(opp_action, rng)
            if not s.is_terminal():
                s = s._apply_my_action(my_action, rng)

        # End of full turn
        if not s.is_terminal():
            s = s._apply_end_of_turn_chip()

        return replace(s, ply=s.ply + 1)

    def _apply_my_action(self, action: Action, rng: random.Random) -> "ShadowState":
        return self._apply_my_switch(action[1]) if action[0] == "switch" else self._apply_my_move(action[1], rng)

    def _apply_opp_action(self, action: Action, rng: random.Random) -> "ShadowState":
        return self._apply_opp_switch(action[1]) if action[0] == "switch" else self._apply_opp_move(action[1], rng)

    def _apply_my_switch(self, new_mon: Any) -> "ShadowState":
        if new_mon is None or is_fainted(new_mon):
            return self
        if self.my_hp.get(id(new_mon), 1.0) <= 0.0:
            return self

        # Keep contexts aligned (best-effort)
        try:
            ctx_me = replace(self.ctx_me, me=new_mon, opp=self.opp_active)
        except Exception:
            ctx_me = self.ctx_me
            try:
                ctx_me.me = new_mon
                ctx_me.opp = self.opp_active
            except Exception:
                pass

        try:
            ctx_opp = replace(self.ctx_opp, opp=new_mon)
        except Exception:
            ctx_opp = self.ctx_opp
            try:
                ctx_opp.opp = new_mon
            except Exception:
                pass

        return replace(self, my_active=new_mon, ctx_me=ctx_me, ctx_opp=ctx_opp)

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
        if new_mon is None or is_fainted(new_mon):
            return self
        if self.opp_hp.get(id(new_mon), 1.0) <= 0.0:
            return self

        try:
            ctx_opp = replace(self.ctx_opp, me=new_mon, opp=self.my_active)
        except Exception:
            ctx_opp = self.ctx_opp
            try:
                ctx_opp.me = new_mon
                ctx_opp.opp = self.my_active
            except Exception:
                pass

        try:
            ctx_me = replace(self.ctx_me, opp=new_mon)
        except Exception:
            ctx_me = self.ctx_me
            try:
                ctx_me.opp = new_mon
            except Exception:
                pass

        return replace(self, opp_active=new_mon, ctx_opp=ctx_opp, ctx_me=ctx_me)

    def _apply_opp_move(self, move: Any, rng: random.Random) -> "ShadowState":
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

        if is_pivot_move(move) and hit:
            target = s._best_opp_switch()
            if target is not None:
                s = s._apply_opp_switch(target)

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
        if not getattr(self, "debug", False):
            return self
        return replace(self, events=self.events + (msg,))