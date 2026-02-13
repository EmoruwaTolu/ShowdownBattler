"""
Comprehensive MCTS Test Suite
- Increased simulations (500)
- Multiple battle scenarios
- Config: Hybrid ON, tau=3.0
"""

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


# Optimal config from testing
OPTIMAL_CONFIG = MCTSConfig(
    num_simulations=500,  # Increased!
    max_depth=3,
    c_puct=1.6,
    seed=42,
    use_hybrid_expansion=True,
)
OPTIMAL_TAU = 3.0


def run_scenario(name, description, our_team, opp_team, active_us, active_opp):
    """Run MCTS on a scenario and display results."""
    print("\n" + "=" * 80)
    print(f"SCENARIO: {name}")
    print("=" * 80)
    print(f"{description}\n")
    
    # Setup battle
    our_team_dict = {mon._identifier_string: mon for mon in our_team}
    opp_team_dict = {mon._identifier_string: mon for mon in opp_team}
    
    battle = create_mock_battle(
        active_identifier=active_us._identifier_string,
        active=active_us,
        opponent_identifier=active_opp._identifier_string,
        opponent=active_opp,
        team=our_team_dict,
        opponent_team=opp_team_dict
    )
    battle.opponent_team_size = 6
    
    ctx_me = EvalContext(me=active_us, opp=active_opp, battle=battle, cache={})
    ctx_opp = EvalContext(me=active_opp, opp=active_us, battle=battle, cache={})
    
    print(f"Position:")
    print(f"  Us: {active_us.species} {active_us.current_hp_fraction:.0%} HP")
    if hasattr(active_us, 'boosts') and active_us.boosts:
        boosts_str = ", ".join([f"{k}:{v:+d}" for k, v in active_us.boosts.items() if v != 0])
        if boosts_str:
            print(f"      Boosts: {boosts_str}")
    
    print(f"  Opp: {active_opp.species} {active_opp.current_hp_fraction:.0%} HP")
    if hasattr(active_opp, 'boosts') and active_opp.boosts:
        boosts_str = ", ".join([f"{k}:{v:+d}" for k, v in active_opp.boosts.items() if v != 0])
        if boosts_str:
            print(f"       Boosts: {boosts_str}")
    
    print(f"\n  Available switches:")
    for mon in our_team:
        if mon != active_us and mon.current_hp_fraction > 0:
            print(f"    - {mon.species} {mon.current_hp_fraction:.0%} HP")
    
    battle.available_moves = list(active_us.moves.values())
    battle.available_switches = [mon for mon in our_team if mon != active_us and mon.current_hp_fraction > 0]
    
    print(f"\n  Running MCTS ({OPTIMAL_CONFIG.num_simulations} simulations)...")
    
    try:
        picked, stats = search(
            battle=battle,
            ctx_me=ctx_me,
            ctx_opp=ctx_opp,
            score_move_fn=score_move,
            score_switch_fn=score_switch,
            dmg_fn=estimate_damage_fraction,
            cfg=OPTIMAL_CONFIG,
            opp_tau=OPTIMAL_TAU,
            return_stats=True,
            return_tree=True,
        )
        
        root = stats['root']
        
        print(f"\n  ✓ Decision: {picked[0].upper()} {getattr(picked[1], 'id', getattr(picked[1], 'species', 'unknown')).upper()}")
        print(f"  Root Q-value: {root.Q:+.3f} ({'winning' if root.Q > 0 else 'losing' if root.Q < -0.1 else 'even'})")
        
        # Show top 5 actions
        print(f"\n  Top 5 actions explored:")
        actions_with_stats = []
        for action, child in root.children.items():
            kind = action[0]
            obj = action[1]
            outcome = action[2] if len(action) > 2 else None  # Check for outcome label
            
            name = getattr(obj, 'id', getattr(obj, 'species', 'unknown'))
            if outcome:
                name = f"{name} [{outcome}]"  # Add outcome label
            
            actions_with_stats.append((kind, name, child.N, child.Q))
        
        actions_with_stats.sort(key=lambda x: x[2], reverse=True)
        
        for i, (kind, name, visits, q) in enumerate(actions_with_stats[:5], 1):
            pct = 100.0 * visits / root.N if root.N > 0 else 0
            indicator = "→" if i == 1 else " "
            print(f"    {indicator} {i}. [{kind.upper():6s}] {name:20s} | N={visits:3d} ({pct:5.1f}%) Q={q:+.3f}")
        
        # Show raw heuristic scores (before MCTS)
        print(f"\n  Heuristic scores (before MCTS search):")
        for action, child in list(root.children.items())[:10]:  # Show top 10
            kind = action[0]
            obj = action[1]
            outcome = action[2] if len(action) > 2 else None
            
            name = getattr(obj, 'id', getattr(obj, 'species', 'unknown'))
            if outcome:
                name = f"{name} [{outcome}]"
            
            # Calculate raw heuristic
            try:
                if kind == "move":
                    h_score = score_move(obj, battle, ctx_me)
                else:
                    h_score = score_switch(obj, battle, ctx_me)
                
                print(f"    [{kind.upper():6s}] {name:25s} | H={h_score:+7.1f} | Prior={child.prior:.4f}")
            except Exception as e:
                print(f"    [{kind.upper():6s}] {name:25s} | H=ERROR ({e})")
        
        return True
        
    except Exception as e:
        print(f"\n  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_scenarios():
    """Run comprehensive test suite."""
    
    print("=" * 80)
    print("COMPREHENSIVE MCTS TEST SUITE")
    print("=" * 80)
    print(f"Config: {OPTIMAL_CONFIG.num_simulations} sims, hybrid={OPTIMAL_CONFIG.use_hybrid_expansion}, tau={OPTIMAL_TAU}")
    
    gen_data = GenData.from_gen(9)
    randbats_data = load_randbats_data()
    
    # =========================================================================
    # SCENARIO 1: Basic Offensive Play
    # =========================================================================
    garchomp, _, _, _, _ = create_pokemon_from_randbats("Garchomp", randbats_data["Garchomp"], gen_data)
    garchomp._identifier_string = "p1: Garchomp"
    garchomp.current_hp_fraction = 0.80
    
    weavile, _, _, _, _ = create_pokemon_from_randbats("Weavile", randbats_data["Weavile"], gen_data)
    weavile._identifier_string = "p1: Weavile"
    weavile.current_hp_fraction = 0.85
    
    landorus, _, _, _, _ = create_pokemon_from_randbats("Landorus-Therian", randbats_data["Landorus-Therian"], gen_data)
    landorus._identifier_string = "p2: Landorus-Therian"
    landorus.current_hp_fraction = 0.65
    
    clefable, _, _, _, _ = create_pokemon_from_randbats("Clefable", randbats_data["Clefable"], gen_data)
    clefable._identifier_string = "p2: Clefable"
    clefable.current_hp_fraction = 0.40
    
    run_scenario(
        "Offensive Play - Type Advantage",
        "Garchomp vs weakened Landorus. Should MCTS go for the KO or set up hazards?",
        [garchomp, weavile],
        [landorus, clefable],
        garchomp,
        landorus
    )
    
    # =========================================================================
    # SCENARIO 2: Setup Sweeper Opportunity
    # =========================================================================
    lucario, _, _, _, _ = create_pokemon_from_randbats("Lucario", randbats_data["Lucario"], gen_data)
    lucario._identifier_string = "p1: Lucario"
    lucario.current_hp_fraction = 0.85
    
    blissey, _, _, _, _ = create_pokemon_from_randbats("Blissey", randbats_data["Blissey"], gen_data)
    blissey._identifier_string = "p2: Blissey"
    blissey.current_hp_fraction = 0.90
    
    tyranitar, _, _, _, _ = create_pokemon_from_randbats("Tyranitar", randbats_data["Tyranitar"], gen_data)
    tyranitar._identifier_string = "p1: Tyranitar"
    tyranitar.current_hp_fraction = 0.70
    
    run_scenario(
        "Setup Opportunity - Swords Dance",
        "Lucario vs Blissey (passive wall). Should MCTS set up with Swords Dance?",
        [lucario, tyranitar],
        [blissey, landorus],
        lucario,
        blissey
    )
    
    # =========================================================================
    # SCENARIO 3: Revenge Killing
    # =========================================================================
    weavile2, _, _, _, _ = create_pokemon_from_randbats("Weavile", randbats_data["Weavile"], gen_data)
    weavile2._identifier_string = "p1: Weavile2"
    weavile2.current_hp_fraction = 1.00  # Full HP
    
    garchomp2, _, _, _, _ = create_pokemon_from_randbats("Garchomp", randbats_data["Garchomp"], gen_data)
    garchomp2._identifier_string = "p2: Garchomp2"
    garchomp2.current_hp_fraction = 0.45  # Weakened
    
    corviknight, _, _, _, _ = create_pokemon_from_randbats("Corviknight", randbats_data["Corviknight"], gen_data)
    corviknight._identifier_string = "p1: Corviknight"
    corviknight.current_hp_fraction = 0.80
    
    run_scenario(
        "Revenge Kill - Ice Shard Priority",
        "Weavile (full HP) vs weakened Garchomp. Ice Shard should KO. Will MCTS use it?",
        [weavile2, corviknight],
        [garchomp2, clefable],
        weavile2,
        garchomp2
    )
    
    # =========================================================================
    # SCENARIO 4: Bad Matchup - Should Switch
    # =========================================================================
    heatran, _, _, _, _ = create_pokemon_from_randbats("Heatran", randbats_data["Heatran"], gen_data)
    heatran._identifier_string = "p1: Heatran"
    heatran.current_hp_fraction = 0.75
    
    swampert, _, _, _, _ = create_pokemon_from_randbats("Swampert", randbats_data["Swampert"], gen_data)
    swampert._identifier_string = "p2: Swampert"
    swampert.current_hp_fraction = 0.90
    
    rillaboom, _, _, _, _ = create_pokemon_from_randbats("Rillaboom", randbats_data["Rillaboom"], gen_data)
    rillaboom._identifier_string = "p1: Rillaboom"
    rillaboom.current_hp_fraction = 0.95
    
    run_scenario(
        "Bad Matchup - Switch Out",
        "Heatran vs Swampert (4x weak to Ground). Should MCTS switch to Rillaboom?",
        [heatran, rillaboom],
        [swampert, garchomp2],
        heatran,
        swampert
    )
    
    # =========================================================================
    # SCENARIO 5: Endgame 1v1
    # =========================================================================
    dragapult, _, _, _, _ = create_pokemon_from_randbats("Dragapult", randbats_data["Dragapult"], gen_data)
    dragapult._identifier_string = "p1: Dragapult"
    dragapult.current_hp_fraction = 0.55
    
    toxapex, _, _, _, _ = create_pokemon_from_randbats("Toxapex", randbats_data["Toxapex"], gen_data)
    toxapex._identifier_string = "p2: Toxapex"
    toxapex.current_hp_fraction = 0.60
    
    run_scenario(
        "Endgame 1v1 - No Switches",
        "Dragapult vs Toxapex. Last mon standing. What's the play?",
        [dragapult],
        [toxapex],
        dragapult,
        toxapex
    )
    
    print("\n" + "=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  • Check if MCTS makes sensible decisions in each scenario")
    print("  • Look at Q-values (positive = winning, negative = losing)")
    print("  • Verify switches are explored when appropriate")
    print("  • Compare exploration % (should focus on best moves)")


if __name__ == "__main__":
    test_all_scenarios()