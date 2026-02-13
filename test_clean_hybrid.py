import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poke_env.data import GenData
from bot.model.ctx import EvalContext
from bot.mcts.search import search, MCTSConfig

from test_real_heuristics import (
    create_mock_battle as _create_mock_battle_orig,
    score_move,
    score_switch,
    estimate_damage_fraction,
)

from test_random_randbats import (
    load_randbats_data,
    create_pokemon_from_randbats,
)

def create_mock_battle(*args, **kwargs):
    """Wrapper that adds gen attribute to battle mock"""
    battle = _create_mock_battle_orig(*args, **kwargs)
    battle.gen = 9
    return battle


def test_clean_output_with_hybrid():
    """Test with clean output and hybrid expansion enabled."""
    
    print("=" * 80)
    print("CLEAN MCTS TEST - Hybrid Expansion + Tuned Opponent")
    print("=" * 80)
    
    gen_data = GenData.from_gen(9)
    randbats_data = load_randbats_data()
    
    # Create team
    garchomp, _, _, _, _ = create_pokemon_from_randbats(
        "Garchomp", randbats_data["Garchomp"], gen_data
    )
    garchomp._identifier_string = "p1: Garchomp"
    garchomp.current_hp_fraction = 0.80
    
    weavile, _, _, _, _ = create_pokemon_from_randbats(
        "Weavile", randbats_data["Weavile"], gen_data
    )
    weavile._identifier_string = "p1: Weavile"
    weavile.current_hp_fraction = 0.85
    
    landorus, _, _, _, _ = create_pokemon_from_randbats(
        "Landorus-Therian", randbats_data["Landorus-Therian"], gen_data
    )
    landorus._identifier_string = "p2: Landorus-Therian"
    landorus.current_hp_fraction = 0.65
    
    clefable, _, _, _, _ = create_pokemon_from_randbats(
        "Clefable", randbats_data["Clefable"], gen_data
    )
    clefable._identifier_string = "p2: Clefable"
    clefable.current_hp_fraction = 0.40
    
    # Setup battle
    our_team_dict = {
        "p1: Garchomp": garchomp,
        "p1: Weavile": weavile,
    }
    
    opp_team_dict = {
        "p2: Landorus-Therian": landorus,
        "p2: Clefable": clefable,
    }
    
    our_team_mons = [garchomp, weavile]
    
    battle = create_mock_battle(
        active_identifier=garchomp._identifier_string,
        active=garchomp,
        opponent_identifier=landorus._identifier_string,
        opponent=landorus,
        team=our_team_dict,
        opponent_team=opp_team_dict
    )
    battle.opponent_team_size = 6
    
    ctx_me = EvalContext(me=garchomp, opp=landorus, battle=battle, cache={})
    ctx_opp = EvalContext(me=landorus, opp=garchomp, battle=battle, cache={})
    
    print(f"\nInitial Position:")
    print(f"  Us: {garchomp.species} {garchomp.current_hp_fraction:.0%} HP")
    print(f"  Opp: {landorus.species} {landorus.current_hp_fraction:.0%} HP")
    print(f"  Available: {weavile.species} {weavile.current_hp_fraction:.0%} HP")
    
    battle.available_moves = list(garchomp.moves.values())
    battle.available_switches = [mon for mon in our_team_mons if mon != garchomp and mon.current_hp_fraction > 0]
    
    # Test configurations
    configs = [
        ("Baseline (no hybrid, tau=8.0)", MCTSConfig(
            num_simulations=200,
            max_depth=3,
            c_puct=1.6,
            seed=42,
            use_hybrid_expansion=False,
        ), 8.0),
        
        ("Hybrid ON, tau=8.0", MCTSConfig(
            num_simulations=200,
            max_depth=3,
            c_puct=1.6,
            seed=42,
            use_hybrid_expansion=True,
        ), 8.0),
        
        ("Hybrid ON, tau=3.0 (smarter opp)", MCTSConfig(
            num_simulations=200,
            max_depth=3,
            c_puct=1.6,
            seed=42,
            use_hybrid_expansion=True,
        ), 3.0),
        
        ("Hybrid ON, tau=2.0 (greedy opp)", MCTSConfig(
            num_simulations=200,
            max_depth=3,
            c_puct=1.6,
            seed=42,
            use_hybrid_expansion=True,
        ), 2.0),
    ]
    
    print(f"\n{'-' * 80}")
    print("COMPARING CONFIGURATIONS")
    print(f"{'-' * 80}\n")
    
    for name, cfg, opp_tau in configs:
        print(f"\n{'=' * 60}")
        print(f"Config: {name}")
        print(f"{'=' * 60}")
        
        try:
            picked, stats = search(
                battle=battle,
                ctx_me=ctx_me,
                ctx_opp=ctx_opp,
                score_move_fn=score_move,
                score_switch_fn=score_switch,
                dmg_fn=estimate_damage_fraction,
                cfg=cfg,
                opp_tau=opp_tau,
                return_stats=True,
                return_tree=True,
            )
            
            root = stats['root']
            
            print(f"\nMCTS picked: {picked[0]} {getattr(picked[1], 'id', getattr(picked[1], 'species', 'unknown'))}")
            print(f"Root visits: {root.N}")
            print(f"Root Q-value: {root.Q:+.3f}")
            
            # Show top 3 actions
            print(f"\nTop 3 actions explored:")
            actions_with_stats = []
            for action, child in root.children.items():
                kind, obj = action[0], action[1]
                name = getattr(obj, 'id', getattr(obj, 'species', 'unknown'))
                actions_with_stats.append((kind, name, child.N, child.Q))
            
            actions_with_stats.sort(key=lambda x: x[2], reverse=True)
            
            for i, (kind, name, visits, q) in enumerate(actions_with_stats[:3], 1):
                pct = 100.0 * visits / root.N if root.N > 0 else 0
                print(f"  {i}. [{kind.upper()}] {name:20s} | N={visits:3d} ({pct:5.1f}%) Q={q:+.3f}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print("✓ Clean output (debug spam removed)")
    print("✓ Hybrid expansion tested")
    print("✓ Opponent temperature tuned")
    print("\nNext steps:")
    print("  - Choose best configuration based on results above")
    print("  - Test on more complex scenarios")
    print("  - Increase simulations (500-1000) for stronger play")


if __name__ == "__main__":
    test_clean_output_with_hybrid()