from poke_env.player import Player
from poke_env.battle import Battle

from bot.model.ctx import EvalContext
from bot.scoring.move_score import score_move
from bot.scoring.switch_score import score_switch
from bot.scoring.helpers import hp_frac

from bot.model.opponent_model import build_opponent_belief
from bot.mcts.search import mcts_pick_action
from bot.scoring.damage_score import estimate_damage_fraction


class AdvancedHeuristicPlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.turn_count = 0
        self.mcts_decisions = []
        self.heuristic_decisions = []
        self.mcts_failures = 0
        
        # Tunable parameters
        self.use_mcts = True
        self.mcts_threshold = 150  # Don't use MCTS for obvious KOs
        self.verbose = True

    def choose_move(self, battle: Battle):
        self.turn_count += 1
        ctx = EvalContext.from_battle(battle)

        actions = [("move", mv) for mv in battle.available_moves] + \
                [("switch", pk) for pk in battle.available_switches]

        if not actions:
            return self.choose_random_move(battle)

        if len(actions) == 1:
            return self.create_order(actions[0][1])

        # Always compute once (used everywhere)
        move_scores = [(mv, float(score_move(mv, battle, ctx))) for mv in battle.available_moves]

        # If MCTS disabled: pure heuristic
        if not self.use_mcts:
            return self.heuristic_fallback(battle, ctx, move_scores)

        # Obvious best move: skip MCTS
        if move_scores:
            best_move, best_score = max(move_scores, key=lambda x: x[1])
            if best_score > self.mcts_threshold:
                if self.verbose:
                    print(f"[Turn {self.turn_count}] Obvious choice: {best_move.id} (score: {best_score:.1f})")
                return self.create_order(best_move)

        # Adaptive iterations
        base_iters = 120
        if self.turn_count <= 3:
            iters = base_iters // 2
        elif self.is_endgame(battle):
            iters = base_iters * 2
        else:
            iters = base_iters

        include_switches = self.should_consider_switches(battle, ctx, move_scores)

        picked = None
        stats = None
        try:
            ctx_opp = EvalContext(
                battle=battle,
                me=battle.opponent_active_pokemon,
                opp=battle.active_pokemon,
                cache={},
            )

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
            )

            if picked:
                self.mcts_decisions.append({"turn": self.turn_count, "action": picked, "stats": stats})

        except Exception as e:
            self.mcts_failures += 1
            if self.verbose:
                print(f"[Turn {self.turn_count}] MCTS failed: {e}")
            picked = None

        if picked is None:
            return self.heuristic_fallback(battle, ctx, move_scores)

        if self.verbose and self.turn_count % 5 == 0 and stats is not None:
            self.print_mcts_analysis(stats)

        kind, obj = picked
        return self.create_order(obj)


    def is_endgame(self, battle: Battle) -> bool:
        """Check if we're in endgame (2 or fewer Pokemon each)"""
        our_alive = sum(1 for p in battle.team.values() if not p.fainted)
        opp_alive = sum(1 for p in battle.opponent_team.values() if not p.fainted)
        return our_alive <= 2 or opp_alive <= 2

    def should_consider_switches(self, battle: Battle, ctx: EvalContext, move_scores: list) -> bool:
        """Determine if switches should be considered in MCTS"""
        # Don't switch if we have a great attacking option
        if move_scores:
            best_score = max(score for _, score in move_scores)
            if best_score > 120:
                return False
        
        # Don't switch if winning hard
        if ctx.me and ctx.opp:
            my_hp = hp_frac(ctx.me)
            opp_hp = hp_frac(ctx.opp)
            if my_hp > opp_hp + 0.3:
                return False
        
        # Otherwise consider switches
        return True

    def print_mcts_analysis(self, stats: dict):
        """Pretty print MCTS analysis"""
        top = stats["top"][:3]
        print(f"\n{'=' * 70}")
        print(f"Turn {self.turn_count} - MCTS Analysis")
        print(f"{'=' * 70}")
        
        total_visits = sum(t['visits'] for t in stats['top'])
        
        for i, t in enumerate(top, 1):
            action_desc = f"{t['kind'].upper()} {t['name']}"
            visits_pct = t['visits'] / total_visits * 100 if total_visits > 0 else 0
            print(f"  {i}. {action_desc:30s} | Visits: {t['visits']:3d} ({visits_pct:5.1f}%) | Q: {t['q']:6.2f}")
        
        print(f"{'=' * 70}\n")

    
    def heuristic_fallback(self, battle: Battle, ctx: EvalContext, move_scores):
        scored = []
        for mv, sc in move_scores:
            scored.append(("move", mv, sc))
        for pk in battle.available_switches:
            scored.append(("switch", pk, float(score_switch(pk, battle, ctx))))

        if not scored:
            return self.choose_random_move(battle)

        scored.sort(key=lambda x: x[2], reverse=True)
        kind, obj, score = scored[0]

        self.heuristic_decisions.append({"turn": self.turn_count, "action": (kind, obj), "score": score})
        return self.create_order(obj)

    
    def battle_finished(self):
        """Analyze battle performance"""
        total = len(self.mcts_decisions) + len(self.heuristic_decisions)
        mcts_pct = len(self.mcts_decisions) / total * 100 if total > 0 else 0
        
        print(f"\n{'=' * 70}")
        print(f"Battle Summary")
        print(f"{'=' * 70}")
        print(f"MCTS decisions: {len(self.mcts_decisions)} ({mcts_pct:.1f}%)")
        print(f"Heuristic fallback: {len(self.heuristic_decisions)}")
        print(f"MCTS failures: {self.mcts_failures}")
        print(f"{'=' * 70}\n")