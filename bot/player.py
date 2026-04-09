from poke_env.player import Player
from poke_env.battle import Battle

from bot.model.ctx import EvalContext
from bot.scoring.move_score import score_move
from bot.scoring.switch_score import score_switch
from bot.scoring.helpers import hp_frac

from bot.model.opponent_model import build_opponent_belief, build_move_pool
from bot.mcts.search import mcts_pick_action
from bot.scoring.damage_score import estimate_damage_fraction
from bot.learning.collect import DataCollector
from bot.learning.policy_model import PolicyModel
from bot.learning.value_model import ValueModel


class AdvancedHeuristicPlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.turn_count = 0
        self.mcts_decisions = []
        self.heuristic_decisions = []
        self.mcts_failures = 0
        self._prev_state = None  # for outcome logging next turn

        # Tunable parameters
        self.use_mcts = True
        self.mcts_threshold = 150  # Don't use MCTS for obvious KOs
        self.verbose = True

        # Data collection for offline learning
        self._collector = DataCollector("data/turns.jsonl")

        # Learned policy model (optional — loaded if weights file exists)
        import os
        policy_path = "data/policy_weights.npz"
        self._policy = PolicyModel.load(policy_path) if os.path.exists(policy_path) else None
        if self._policy is not None and self.verbose:
            print(f"  [policy] Loaded learned priors from {policy_path}")

        # Learned value model (optional — loaded if weights file exists)
        value_path = "data/value_weights.npz"
        self._value = ValueModel.load(value_path) if os.path.exists(value_path) else None
        if self._value is not None and self.verbose:
            print(f"  [value]  Loaded learned value fn from {value_path}")

        # Mixing weight for learned prior; overridable via selfplay.py --policy-alpha
        self.policy_alpha: float = 0.3

    MAX_BATTLE_TURNS = 80

    async def choose_move(self, battle: Battle):
        self.turn_count += 1

        if battle.turn > self.MAX_BATTLE_TURNS:
            if self.verbose:
                print(f"  [selfplay] Turn {battle.turn} exceeds MAX_BATTLE_TURNS={self.MAX_BATTLE_TURNS} — forfeiting")
            await self.ps_client.send_message("/forfeit", battle.battle_tag)
            return self.choose_random_move(battle)

        ctx = EvalContext.from_battle(battle)

        actions = [("move", mv) for mv in battle.available_moves] + \
                [("switch", pk) for pk in battle.available_switches]

        if not actions:
            return self.choose_random_move(battle)

        if len(actions) == 1:
            return self.create_order(actions[0][1])

        # Always compute heuristic scores (used for priors + display)
        move_scores = [(mv, float(score_move(mv, battle, ctx))) for mv in battle.available_moves]
        switch_scores = [(pk, float(score_switch(pk, battle, ctx))) for pk in battle.available_switches]

        # Build opponent beliefs, move pools and ctx early — used by display and MCTS
        gen = getattr(battle, 'gen', 9) or 9
        opp_beliefs = {}
        opp_move_pools = {}
        for p in battle.opponent_team.values():
            if p is not None:
                belief = build_opponent_belief(p, gen)
                opp_beliefs[id(p)] = belief
                opp_move_pools[id(p)] = build_move_pool(belief, gen)

        # Update beliefs from last turn's outcome
        self._apply_belief_observations(battle, opp_beliefs)

        ctx_opp = EvalContext(
            battle=battle,
            me=battle.opponent_active_pokemon,
            opp=battle.active_pokemon,
            cache={},
        )

        if self.verbose:
            self._print_turn_header(battle, ctx)
            self._print_opp_prediction(battle, ctx_opp, opp_beliefs, opp_move_pools)

        # If MCTS disabled: pure heuristic
        if not self.use_mcts:
            return self.heuristic_fallback(battle, ctx, move_scores, switch_scores)

        # Obvious best move: skip MCTS
        if move_scores:
            best_move, best_score = max(move_scores, key=lambda x: x[1])
            if best_score > self.mcts_threshold:
                if self.verbose:
                    self._print_options(battle, move_scores, switch_scores,
                                        stats=None, picked=("move", best_move))
                    print(f"  >> OBVIOUS: {best_move.id} (heuristic {best_score:.1f} > {self.mcts_threshold})\n")
                self._store_prev_state(battle, "move", best_move)
                return self.create_order(best_move)

        # Adaptive iterations
        base_iters = 300
        if self.turn_count <= 3:
            iters = base_iters // 2
        elif self.is_endgame(battle):
            iters = base_iters * 2
        else:
            iters = base_iters

        include_switches = self.should_consider_switches(battle, ctx, move_scores)

        if self.verbose:
            print(f"  Running MCTS ({iters} sims, switches={'yes' if include_switches else 'no'}) ...")

        picked = None
        stats = None
        try:
            picked, stats = mcts_pick_action(
                battle=battle,
                ctx=ctx,
                ctx_opp=ctx_opp,
                score_move_fn=score_move,
                score_switch_fn=score_switch,
                dmg_fn=estimate_damage_fraction,
                iters=iters,
                max_depth=4,
                include_switches=include_switches,
                opp_beliefs=opp_beliefs,
                opp_move_pools=opp_move_pools,
                policy_model=self._policy,
                policy_alpha=self.policy_alpha,
                value_model=self._value,
            )

            if picked:
                self.mcts_decisions.append({"turn": self.turn_count, "action": picked, "stats": stats})

        except Exception as e:
            self.mcts_failures += 1
            if self.verbose:
                import traceback
                print(f"  MCTS failed: {e}")
                traceback.print_exc()
            picked = None

        if picked is None or picked[1] is None:
            if self.verbose:
                print(f"  >> HEURISTIC FALLBACK\n")
            return self.heuristic_fallback(battle, ctx, move_scores, switch_scores)

        if self.verbose:
            self._print_options(battle, move_scores, switch_scores,
                                stats=stats, picked=picked,
                                include_switches_in_mcts=include_switches)
            kind, obj = picked[0], picked[1]
            chosen_name = getattr(obj, "id", None) or getattr(obj, "species", str(obj))
            print(f"  >> MCTS chose: {kind.upper()} {chosen_name}\n")

        # Record decision for offline learning
        try:
            eval_terms = stats.get("eval_terms", {}) if stats else {}
            eval_value = float(stats.get("eval_value", 0.0)) if stats else 0.0
            self._collector.record_turn(
                battle_id=battle.battle_tag,
                turn=battle.turn,
                eval_terms=eval_terms,
                eval_value=eval_value,
                mcts_stats=stats,
                opp_beliefs=opp_beliefs,
                opp_active=battle.opponent_active_pokemon,
                ctx_me=ctx,
                battle=battle,
                picked=picked,
                shadow_state=stats.get("root_state") if stats else None,
            )
        except Exception as _rec_err:
            if self.verbose:
                print(f"  [collect] record_turn failed: {_rec_err}")

        self._store_prev_state(battle, picked[0], picked[1])
        return self.create_order(picked[1])


    def _store_prev_state(self, battle, kind, obj):
        """Store end-of-turn state so next turn can log the outcome and update beliefs."""
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        name = getattr(obj, "id", None) or getattr(obj, "species", str(obj))

        # Capture move info for observe_damage_taken next turn
        move_bp, move_is_special, atk_stat = None, None, None
        if kind == "move":
            try:
                move_bp = int(getattr(obj, "base_power", 0) or 0)
                cat = getattr(getattr(obj, "type", None), "name", None)
                # category: PHYSICAL/SPECIAL/STATUS
                cat_str = str(getattr(obj, "category", "")).upper()
                move_is_special = (cat_str == "SPECIAL")
                if me and move_bp > 0:
                    bs = getattr(me, "stats", {}) or {}
                    atk_stat = int(bs.get("spa" if move_is_special else "atk", 100))
                    # Apply stat boosts
                    boosts = getattr(me, "boosts", {}) or {}
                    stage = int(boosts.get("spa" if move_is_special else "atk", 0))
                    if stage >= 0:
                        atk_stat = int(atk_stat * (2 + stage) / 2)
                    else:
                        atk_stat = int(atk_stat * 2 / (2 - stage))
            except Exception:
                move_bp, move_is_special, atk_stat = None, None, None

        self._prev_state = {
            "turn": self.turn_count,
            "action_str": f"{kind.upper()} {name}",
            "kind": kind,
            "my_species": getattr(me, "species", "?") if me else "?",
            "my_hp": hp_frac(me) if me else 1.0,
            "opp_species": getattr(opp, "species", "?") if opp else "?",
            "opp_hp": hp_frac(opp) if opp else 1.0,
            "opp_id": id(opp) if opp else None,
            # Move info for observe_damage_taken
            "move_bp": move_bp,
            "move_is_special": move_is_special,
            "atk_stat": atk_stat,
        }

    def _apply_belief_observations(self, battle, opp_beliefs: dict) -> None:
        """
        After each turn, update the active opponent's belief based on:

        1. observe_damage_taken — if we used a damaging move last turn and the same
           opponent is still active, the HP delta gives us a damage fraction signal.

        2. observe_speed_comparison — if we can determine who moved first from the
           previous turn's event log, penalise candidates whose speed contradicts it.

        Both calls are best-effort and silently skip on any exception.
        """
        prev = self._prev_state
        if prev is None:
            return

        opp = battle.opponent_active_pokemon
        if opp is None:
            return

        # Only update the belief for the same Pokemon that was active last turn
        if getattr(opp, "species", None) != prev.get("opp_species"):
            return

        belief = opp_beliefs.get(id(opp))
        if belief is None:
            return

        # 1. observe_damage_taken
        if (prev.get("kind") == "move"
                and prev.get("move_bp") and prev["move_bp"] > 0
                and prev.get("atk_stat") and prev["atk_stat"] > 0
                and prev.get("move_is_special") is not None):
            opp_hp_now = hp_frac(opp)
            opp_hp_then = float(prev.get("opp_hp", 1.0))
            damage_frac = opp_hp_then - opp_hp_now
            # Only call when damage is plausibly from our attack (>1% and <100%)
            if 0.01 < damage_frac < 1.0:
                try:
                    belief.observe_damage_taken(
                        base_power=int(prev["move_bp"]),
                        is_special=bool(prev["move_is_special"]),
                        attacker_stat=int(prev["atk_stat"]),
                        damage_fraction=float(damage_frac),
                    )
                except Exception:
                    pass

        # 2. observe_speed_comparison
        moved_first = self._who_moved_first(battle)
        if moved_first != 0:
            try:
                me = battle.active_pokemon
                my_spe = 0
                if me is not None:
                    bs = getattr(me, "stats", {}) or {}
                    my_spe = int(bs.get("spe", 80))
                    # Apply speed boosts
                    boosts = getattr(me, "boosts", {}) or {}
                    stage = int(boosts.get("spe", 0))
                    if stage >= 0:
                        my_spe = int(my_spe * (2 + stage) / 2)
                    else:
                        my_spe = int(my_spe * 2 / (2 - stage))
                    # Paralysis halves speed
                    from poke_env.battle import Status
                    if getattr(me, "status", None) == Status.PAR:
                        my_spe = my_spe // 2
                    # Choice Scarf
                    item = str(getattr(me, "item", "") or "").lower().replace(" ", "").replace("-", "")
                    if item == "choicescarf":
                        my_spe = int(my_spe * 1.5)

                if my_spe > 0:
                    from poke_env.battle import Field
                    trick_room = Field.TRICK_ROOM in (getattr(battle, "fields", {}) or {})
                    belief.observe_speed_comparison(
                        our_speed=my_spe,
                        moved_first=(moved_first == 1),
                        trick_room_active=trick_room,
                    )
            except Exception:
                pass

    @staticmethod
    def _who_moved_first(battle) -> int:
        """
        Parse the previous turn's event log to determine who moved first.

        Returns:
          +1  if we moved first (our |move| appeared before the opponent's)
          -1  if the opponent moved first
           0  if undetermined (switch, no events, force-switch, etc.)
        """
        prev_turn = battle.turn - 1
        if prev_turn < 1:
            return 0
        obs = battle.observations.get(prev_turn)
        if obs is None:
            return 0

        my_role = getattr(battle, "_player_role", None)  # "p1" or "p2"
        if my_role is None:
            return 0

        for event in obs.events:
            if len(event) >= 3 and event[1] == "move":
                actor = event[2]  # e.g. "p1a: Gardevoir"
                if actor.startswith(my_role):
                    return +1
                else:
                    return -1
        return 0  # no |move| events this turn (both switched, or force-switch)

    @staticmethod
    def _status_str(pkmn):
        """Status tag, with toxic-turn counter for TOX."""
        status = getattr(pkmn, "status", None)
        if not status:
            return ""
        name = status.name
        if name == "TOX":
            ctr = getattr(pkmn, "status_counter", 0) or 0
            return f" [TOX:{ctr}]"
        return f" [{name}]"

    @staticmethod
    def _boosts_str(pkmn):
        """Non-zero stat stages as '+2Atk -1Def …'."""
        boosts = getattr(pkmn, "boosts", {}) or {}
        labels = {"atk": "Atk", "def": "Def", "spa": "SpA", "spd": "SpD",
                  "spe": "Spe", "accuracy": "Acc", "evasion": "Eva"}
        parts = [f"{v:+d}{labels.get(k, k)}" for k, v in boosts.items() if v != 0]
        return " ".join(parts)

    @staticmethod
    def _volatile_str(pkmn):
        """Compact list of notable volatile conditions."""
        effects = getattr(pkmn, "effects", {}) or {}
        _SHOW = {
            "CONFUSION": "Confuse", "LEECHSEED": "LSeed",
            "SUBSTITUTE": "Sub",    "YAWN": "Yawn",
            "ATTRACT": "Attract",   "PERISHSONG": "Perish",
            "TAUNT": "Taunt",       "ENCORE": "Encore",
            "TRAPPED": "Trapped",   "PARTIALLYTRAPPED": "Bound",
        }
        parts = []
        for eff, val in effects.items():
            key = getattr(eff, "name", str(eff)).upper().replace("_", "")
            label = _SHOW.get(key)
            if label:
                parts.append(f"{label}:{val}" if isinstance(val, int) and val > 1 else label)
        return " ".join(parts)

    @staticmethod
    def _side_cond_str(conditions):
        """Abbreviate side-condition names."""
        _ABBR = {
            "STEALTHROCK": "SR", "SPIKES": "Spikes", "TOXICSPIKES": "TSpikes",
            "STICKYWEB": "Web", "REFLECT": "Reflect", "LIGHTSCREEN": "LightScr",
            "AURORAVEIL": "AuroraV", "TAILWIND": "Tailwind", "LUCKYCHANT": "LuckyC",
        }
        parts = []
        for sc, val in (conditions or {}).items():
            key = getattr(sc, "name", str(sc)).upper().replace("_", "")
            label = _ABBR.get(key, key)
            parts.append(f"{label}x{val}" if isinstance(val, int) and val > 1 else label)
        return " ".join(parts)

    @staticmethod
    def _weather_terrain_str(battle):
        """One-line summary of active weather and terrain."""
        _W = {"SUNNYDAY": "Sun", "RAINDANCE": "Rain", "SANDSTORM": "Sand",
              "SNOW": "Snow", "HAIL": "Hail", "DESOLATELAND": "HarshSun",
              "PRIMORDIALSEA": "HeavyRain", "DELTASTREAM": "StrongWind"}
        _T = {"ELECTRICTERRAIN": "E-Terrain", "GRASSYTERRAIN": "G-Terrain",
              "MISTYTERRAIN": "M-Terrain",   "PSYCHICTERRAIN": "P-Terrain"}
        parts = []
        for w in (getattr(battle, "weather", {}) or {}):
            parts.append(_W.get(getattr(w, "name", str(w)).upper().replace("_", ""), w.name))
        for f in (getattr(battle, "fields", {}) or {}):
            parts.append(_T.get(getattr(f, "name", str(f)).upper().replace("_", ""), f.name))
        return " | ".join(parts)

    @staticmethod
    def _print_belief_state(battle):
        """Print per-Pokemon role/move-pool beliefs for every revealed opponent."""
        gen = getattr(battle, "gen", 9) or 9
        seen = [p for p in battle.opponent_team.values() if p is not None]
        if not seen:
            return

        print("  Beliefs:")
        for p in seen:
            species = getattr(p, "species", "?")
            hp_now = f"{hp_frac(p)*100:.0f}%"
            fainted = getattr(p, "fainted", False)
            tag = " [FNT]" if fainted else ""

            try:
                belief = build_opponent_belief(p, gen)
            except Exception:
                print(f"    {species}{tag} — (belief error)")
                continue

            n_roles = len(belief.dist)

            # Confirmed observations
            obs = []
            if belief.revealed_moves:
                obs.append("moves:[" + " ".join(sorted(belief.revealed_moves)) + "]")
            if belief.revealed_item:
                obs.append(f"item:{belief.revealed_item}")
            if belief.revealed_ability:
                obs.append(f"abil:{belief.revealed_ability}")
            obs_str = "  " + "  ".join(obs) if obs else ""

            print(f"    {species} {hp_now}{tag} | {n_roles} role(s){obs_str}")

            # Show top 3 roles with their move pools
            top = sorted(belief.dist, key=lambda x: x[1], reverse=True)[:3]
            for cand, prob in top:
                role_label = cand.id.split(":", 1)[1] if ":" in cand.id else cand.id
                # Confirmed moves marked *, rest are pool
                confirmed = sorted(belief.revealed_moves & cand.moves)
                pool      = sorted(cand.moves - belief.revealed_moves)
                confirmed_str = (" ".join(f"{m}*" for m in confirmed)) if confirmed else ""
                pool_str      = (" ".join(pool[:8])) if pool else "—"
                moves_str = (confirmed_str + ("  " if confirmed_str and pool_str != "—" else "") + pool_str).strip()
                print(f"      {prob*100:5.1f}%  {role_label:<22}  {moves_str}")

    @staticmethod
    def _print_opp_prediction(battle, ctx_opp, opp_beliefs, opp_move_pools):
        """
        Print expected opponent action distribution at root.

        For each move in the belief move pool, weight by the sum of role
        probabilities that include it, score with ctx_opp, then softmax(tau=8).
        Switches to known bench are included unweighted.
        """
        import math

        opp = battle.opponent_active_pokemon
        if opp is None:
            return

        opp_id = id(opp)
        belief    = opp_beliefs.get(opp_id)
        move_pool = opp_move_pools.get(opp_id) if opp_move_pools else None

        # Collect moves: move_id -> (Move object, prior weight)
        move_weights: dict = {}
        if belief and move_pool:
            for cand, cand_prob in belief.dist:
                for mid in cand.moves:
                    mv = move_pool.get(mid)
                    if mv is not None:
                        if mid not in move_weights:
                            move_weights[mid] = [mv, 0.0]
                        move_weights[mid][1] += cand_prob
        else:
            for mv in (getattr(opp, "moves", None) or {}).values():
                mid = str(getattr(mv, "id", ""))
                if mid:
                    move_weights[mid] = [mv, 1.0]

        if not move_weights:
            return

        # Score each move from the opponent's perspective
        candidates = []   # (label, raw_score, prior_weight)
        for mid, (mv, weight) in move_weights.items():
            try:
                s = float(score_move(mv, battle, ctx_opp))
                candidates.append((mid, s, weight))
            except Exception:
                pass

        # Known bench switches
        for p in battle.opponent_team.values():
            if p is opp or p is None or getattr(p, "fainted", False):
                continue
            try:
                s = float(score_switch(p, battle, ctx_opp))
                candidates.append((f"~{getattr(p, 'species', '?')}", s, 1.0))
            except Exception:
                pass

        if not candidates:
            return

        # Softmax(tau=8) then weight by prior, renormalise
        tau = 8.0
        scores = [s for _, s, _ in candidates]
        m = max(scores)
        exps = [math.exp((s - m) / tau) for s in scores]
        z = sum(exps) or 1.0
        softmax_probs = [e / z for e in exps]

        weighted = [sp * w for sp, (_, _, w) in zip(softmax_probs, candidates)]
        wz = sum(weighted) or 1.0
        final_probs = [wp / wz for wp in weighted]

        ranked = sorted(zip([c[0] for c in candidates], final_probs),
                        key=lambda x: x[1], reverse=True)

        top = [(name, p) for name, p in ranked if p >= 0.02][:6]
        if top:
            print("  Opp expected : " + "  ".join(f"{n}({p*100:.0f}%)" for n, p in top))

    def _print_turn_header(self, battle, ctx):
        W = 72
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        my_hp  = f"{hp_frac(me)*100:.0f}%" if me else "?%"
        opp_hp = f"{hp_frac(opp)*100:.0f}%" if opp else "?%"
        my_name  = getattr(me,  "species", "?")
        opp_name = getattr(opp, "species", "?")
        opp_types = "/".join(t.name for t in (getattr(opp, "types", None) or []) if t) or "?"

        my_alive  = sum(1 for p in battle.team.values() if not p.fainted)

        opp_seen_alive = sum(1 for p in battle.opponent_team.values() if not p.fainted)
        opp_total = 6
        opp_unseen = max(0, opp_total - len(battle.opponent_team))
        if opp_unseen:
            opp_alive_str = f"{opp_seen_alive} seen + {opp_unseen} unseen"
        else:
            opp_alive_str = f"{opp_seen_alive} alive"

        print(f"\n{'─'*W}")
        print(f"  Turn {self.turn_count}  |  "
              f"US: {my_name} {my_hp}{self._status_str(me)} ({my_alive} alive)  "
              f"vs  OPP: {opp_name}[{opp_types}] {opp_hp}{self._status_str(opp)} ({opp_alive_str})")

        # Stat boosts
        my_b  = self._boosts_str(me)
        opp_b = self._boosts_str(opp)
        if my_b or opp_b:
            print(f"  Boosts : US [{my_b or '—'}]  OPP [{opp_b or '—'}]")

        # Volatile conditions
        my_v  = self._volatile_str(me)
        opp_v = self._volatile_str(opp)
        if my_v or opp_v:
            print(f"  Volatile: US [{my_v or '—'}]  OPP [{opp_v or '—'}]")

        # Entry hazards / screens on each side
        my_side  = self._side_cond_str(getattr(battle, "side_conditions", {}))
        opp_side = self._side_cond_str(getattr(battle, "opponent_side_conditions", {}))
        if my_side or opp_side:
            print(f"  Hazards: US [{my_side or '—'}]  OPP [{opp_side or '—'}]")

        # Weather / terrain
        wt = self._weather_terrain_str(battle)
        if wt:
            print(f"  Field  : {wt}")

        print(f"{'─'*W}")

        # Log outcome of the previous turn
        if self._prev_state is not None:
            self._print_last_turn_outcome(battle)

        # Opponent belief state
        self._print_belief_state(battle)

    def _print_last_turn_outcome(self, battle):
        """One-liner showing what changed since we made our last decision."""
        prev = self._prev_state
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        my_species_now  = getattr(me,  "species", "?") if me  else "?"
        opp_species_now = getattr(opp, "species", "?") if opp else "?"
        my_hp_now  = hp_frac(me)  if me  else 0.0
        opp_hp_now = hp_frac(opp) if opp else 0.0

        parts = [f"T{prev['turn']}: {prev['action_str']}"]

        # Our side
        if my_species_now == prev["my_species"]:
            d = my_hp_now - prev["my_hp"]
            if abs(d) > 0.005:
                parts.append(f"US {prev['my_hp']*100:.0f}%→{my_hp_now*100:.0f}% ({d*100:+.0f}%)")
        else:
            parts.append(f"US {prev['my_species']}→{my_species_now}")

        # Opponent side
        if opp_species_now == prev["opp_species"]:
            d = opp_hp_now - prev["opp_hp"]
            if abs(d) > 0.005:
                parts.append(f"OPP {prev['opp_hp']*100:.0f}%→{opp_hp_now*100:.0f}% ({d*100:+.0f}%)")
        else:
            prev_fainted = any(
                p.fainted and getattr(p, "species", "").lower() == prev["opp_species"].lower()
                for p in battle.opponent_team.values()
            )
            suffix = "FAINTED" if prev_fainted else "switched"
            parts.append(f"OPP {prev['opp_species']} {suffix}→{opp_species_now}")

        print("  [" + " | ".join(parts) + "]")

    def _print_options(self, battle, move_scores, switch_scores,
                       stats=None, picked=None, include_switches_in_mcts=True):
        """
        Unified table of every available action.

        Columns (when MCTS ran):
          Action  Type  BP  Acc  Heur  |  N   %    Q    Prior
        Without MCTS:
          Action  Type  BP  Acc  Heur
        The chosen action is marked with <<.
        """
        # Build MCTS lookup: (kind, canonical_name) -> row dict
        mcts = {}
        total_visits = 1
        if stats:
            for row in stats.get("top", []):
                mcts[(row["kind"], row["name"])] = row
            v = sum(r["visits"] for r in stats["top"])
            if v > 0:
                total_visits = v

        has_mcts = bool(mcts)

        def _marker(kind, obj):
            if picked is None:
                return ""
            pk_kind, pk_obj = picked[0], picked[1]
            if pk_kind != kind:
                return ""
            if kind == "move" and getattr(pk_obj, "id", None) == getattr(obj, "id", None):
                return " <<"
            if kind == "switch" and getattr(pk_obj, "species", None) == getattr(obj, "species", None):
                return " <<"
            return ""

        # Sort: by MCTS visits when available, else heuristic score
        def mv_key(x):
            return mcts.get(("move", x[0].id), {}).get("visits", 0) if has_mcts else x[1]
        def sw_key(x):
            return mcts.get(("switch", x[0].species), {}).get("visits", 0) if has_mcts else x[1]

        sorted_moves = sorted(move_scores, key=mv_key, reverse=True)
        sorted_sw    = sorted(switch_scores, key=sw_key, reverse=True)

        if has_mcts:
            print(f"  {'Action':<22} {'Type':>8} {'BP':>3}  {'Acc':>4}  {'Heur':>6}  {'N':>4} {'%':>5}  {'Q':>7}  {'Prior':>5}")
            print(f"  {'─'*76}")
        else:
            print(f"  {'Action':<22} {'Type':>8} {'BP':>3}  {'Acc':>4}  {'Heur':>6}")
            print(f"  {'─'*52}")

        for mv, sc in sorted_moves:
            bp      = getattr(mv, "base_power", 0) or 0
            acc     = getattr(mv, "accuracy", 1.0)
            mv_type = getattr(getattr(mv, "type", None), "name", "?")[:8]
            acc_str = f"{int(acc*100)}%" if isinstance(acc, float) and acc < 1.0 else "—"
            m = _marker("move", mv)
            if has_mcts:
                row = mcts.get(("move", mv.id), {})
                n   = row.get("visits", 0)
                q   = row.get("q", 0.0)
                pr  = row.get("prior", 0.0)
                pct = n / total_visits * 100
                print(f"  {mv.id:<22} {mv_type:>8} {bp:>3d}  {acc_str:>4}  {sc:>6.1f}  {n:>4d} {pct:>4.1f}%  {q:>+7.3f}  {pr:>5.3f}{m}")
            else:
                print(f"  {mv.id:<22} {mv_type:>8} {bp:>3d}  {acc_str:>4}  {sc:>6.1f}{m}")

        for pk, sc in sorted_sw:
            pk_hp   = f"{hp_frac(pk)*100:.0f}%"
            pk_types = "/".join(t.name[:5] for t in (getattr(pk, "types", None) or []) if t)[:8] or "?"
            name    = f"~{pk.species}"
            m = _marker("switch", pk)
            no_mcts_note = "" if include_switches_in_mcts else "*"
            if has_mcts:
                row = mcts.get(("switch", pk.species), {})
                n   = row.get("visits", 0)
                q   = row.get("q", 0.0)
                pr  = row.get("prior", 0.0)
                pct = n / total_visits * 100
                print(f"  {name:<22} {pk_types:>8} {'—':>3}  {pk_hp:>4}  {sc:>6.1f}  {n:>4d} {pct:>4.1f}%  {q:>+7.3f}  {pr:>5.3f}{m}{no_mcts_note}")
            else:
                print(f"  {name:<22} {pk_types:>8} {'—':>3}  {pk_hp:>4}  {sc:>6.1f}{m}")

    def is_endgame(self, battle: Battle) -> bool:
        our_alive = sum(1 for p in battle.team.values() if not p.fainted)
        opp_alive = sum(1 for p in battle.opponent_team.values() if not p.fainted)
        return our_alive <= 2 or opp_alive <= 2

    def should_consider_switches(self, battle: Battle, ctx: EvalContext, move_scores: list) -> bool:
        if move_scores:
            best_score = max(score for _, score in move_scores)
            if best_score > 120:
                return False
        if ctx.me and ctx.opp:
            my_hp = hp_frac(ctx.me)
            opp_hp = hp_frac(ctx.opp)
            if my_hp > opp_hp + 0.3:
                return False
        return True

    def heuristic_fallback(self, battle: Battle, ctx: EvalContext, move_scores, switch_scores=None):
        if switch_scores is None:
            switch_scores = [(pk, float(score_switch(pk, battle, ctx))) for pk in battle.available_switches]

        scored = []
        for mv, sc in move_scores:
            scored.append(("move", mv, sc))
        for pk, sc in switch_scores:
            scored.append(("switch", pk, sc))

        if not scored:
            return self.choose_random_move(battle)

        scored.sort(key=lambda x: x[2], reverse=True)
        kind, obj, score = scored[0]

        if self.verbose:
            self._print_options(battle, move_scores, switch_scores,
                                stats=None, picked=(kind, obj))
            name = getattr(obj, "id", None) or getattr(obj, "species", str(obj))
            print(f"  >> HEURISTIC: {kind.upper()} {name} (score {score:.1f})\n")

        self.heuristic_decisions.append({"turn": self.turn_count, "action": (kind, obj), "score": score})
        self._store_prev_state(battle, kind, obj)
        return self.create_order(obj)

    def _battle_finished_callback(self, battle: Battle):
        """Called by poke_env when a battle ends."""
        total = len(self.mcts_decisions) + len(self.heuristic_decisions)
        mcts_pct = len(self.mcts_decisions) / total * 100 if total > 0 else 0

        print(f"\n{'='*72}")
        print(f"Battle finished — {'WIN' if battle.won else 'LOSS'}")
        print(f"  MCTS decisions  : {len(self.mcts_decisions)} ({mcts_pct:.1f}%)")
        print(f"  Heuristic       : {len(self.heuristic_decisions)}")
        print(f"  MCTS failures   : {self.mcts_failures}")
        print(f"{'='*72}\n")

        # Stamp win/loss onto all buffered records for this battle and flush to disk
        try:
            self._collector.finish_battle(battle.battle_tag, won=battle.won)
            print(f"  [collect] wrote {battle.battle_tag} → {self._collector._path}")
        except Exception as _flush_err:
            print(f"  [collect] finish_battle failed: {_flush_err}")
