from poke_env.player import Player
from poke_env.battle import Battle

from bot.model.ctx import EvalContext
from bot.scoring.move_score import score_move
from bot.scoring.switch_score import score_switch
from bot.scoring.helpers import hp_frac

from bot.model.opponent_model import build_opponent_belief
from bot.mcts.search import mcts_pick_action


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

        # Build action list
        actions = []
        for mv in battle.available_moves:
            actions.append(("move", mv))
        for pk in battle.available_switches:
            actions.append(("switch", pk))

        if not actions:
            return self.choose_random_move(battle)

        # Fast path: only 1 option
        if len(actions) == 1:
            return self.create_order(actions[0][1])

        # Fast path: check for obvious decisions (skip MCTS)
        if not self.use_mcts or len(battle.available_moves) > 0:
            move_scores = [(mv, score_move(mv, battle, ctx)) for mv in battle.available_moves]
            if move_scores:
                best_move, best_score = max(move_scores, key=lambda x: x[1])
                if best_score > self.mcts_threshold:
                    if self.verbose:
                        print(f"[Turn {self.turn_count}] Obvious choice: {best_move.id} (score: {best_score:.1f})")
                    return self.create_order(best_move)

        # Build opponent belief
        belief = None
        try:
            gen = getattr(battle, "gen", 9) or 9
            if ctx.opp is not None:
                belief = build_opponent_belief(ctx.opp, gen=gen)
        except Exception:
            belief = None

        # Determine iteration count (adaptive)
        base_iters = 120
        if self.turn_count <= 3:
            iters = base_iters // 2  # Early game: less time
        elif self._is_endgame(battle):
            iters = base_iters * 2  # Endgame: more time
        else:
            iters = base_iters

        # Determine if we should consider switches
        include_switches = self._should_consider_switches(battle, ctx, move_scores if 'move_scores' in locals() else [])

        # Run MCTS
        picked = None
        stats = None
        try:
            picked, stats = mcts_pick_action(
                battle=battle,
                ctx=ctx,
                belief=belief,
                actions=actions,
                iters=iters,
                max_depth=4,
                include_switches=include_switches,
            )
            
            if picked:
                self.mcts_decisions.append({
                    'turn': self.turn_count,
                    'action': picked,
                    'stats': stats,
                })
                
        except Exception as e:
            self.mcts_failures += 1
            if self.verbose:
                print(f"[Turn {self.turn_count}] MCTS failed: {e}")
            picked = None

        # Fallback to heuristic if MCTS failed
        if picked is None:
            scored = []
            for mv in battle.available_moves:
                scored.append(("move", mv, score_move(mv, battle, ctx)))
            for pk in battle.available_switches:
                scored.append(("switch", pk, score_switch(pk, battle, ctx)))
            
            if not scored:
                return self.choose_random_move(battle)
            
            scored.sort(key=lambda x: x[2], reverse=True)
            kind, obj, score = scored[0]
            
            self.heuristic_decisions.append({
                'turn': self.turn_count,
                'action': (kind, obj),
                'score': score,
            })
            
            return self.create_order(obj)

        # Debug output
        if self.verbose and self.turn_count % 5 == 0 and stats is not None:
            self._print_mcts_analysis(stats)

        kind, obj = picked
        return self.create_order(obj)

    def _is_endgame(self, battle: Battle) -> bool:
        """Check if we're in endgame (2 or fewer Pokemon each)"""
        our_alive = sum(1 for p in battle.team.values() if not p.fainted)
        opp_alive = sum(1 for p in battle.opponent_team.values() if not p.fainted)
        return our_alive <= 2 or opp_alive <= 2

    def _should_consider_switches(self, battle: Battle, ctx: EvalContext, move_scores: list) -> bool:
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

    def _print_mcts_analysis(self, stats: dict):
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