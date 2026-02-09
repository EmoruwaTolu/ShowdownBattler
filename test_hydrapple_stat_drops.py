import sys
import os
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List
from unittest.mock import Mock

from poke_env.battle import MoveCategory, PokemonType, Status
from poke_env.data import to_id_str, GenData
from bot.model.ctx import EvalContext
from bot.mcts.search import search, MCTSConfig

from test_real_heuristics import (
    create_mock_pokemon,
    create_mock_move,
    create_mock_battle,
    score_move,
    score_switch,
    estimate_damage_fraction,
)
from visualize_tree_depth import visualize_tree_depth

# Import helper functions from test_random_randbats
from test_random_randbats import (
    load_randbats_data,
    get_pokemon_info,
    get_move_info,
    get_pokemon_types,
    get_pokemon_base_stats,
    calculate_stat,
    calculate_hp,
    get_move_type,
    get_move_category,
    get_move_power,
    get_move_accuracy,
    get_move_crit_ratio,
    get_move_status,
    get_move_secondary,
    create_pokemon_from_randbats,
)


def test_hydrapple_stat_drops():
    """
    Test scenario: Hydrapple with stat-dropping moves
    
    Tests that we properly track stat drops from:
    - Draco Meteor (guaranteed -2 SpA after use)
    - Leaf Storm (guaranteed -2 SpA after use)
    
    We should see:
    1. Initial state: boosts = {'spa': 0}
    2. After Draco Meteor: boosts = {'spa': -2}
    3. After another use: boosts = {'spa': -4}
    4. After third use: boosts = {'spa': -6} (clamped at -6)
    
    MCTS should understand:
    - First Draco Meteor is strong
    - Second use is weaker (due to -2 SpA)
    - Eventually should switch or use other moves
    """
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "STAT DROP TRACKING TEST" + " " * 35 + "║")
    print("║" + " " * 18 + "HYDRAPPLE VS TYRANITAR" + " " * 37 + "║")
    print("╚" + "=" * 78 + "╝")
    
    gen_data = GenData.from_gen(9)
    randbats_data = load_randbats_data()
    
    player_species = "Hydrapple"
    opponent_species = "Tyranitar"
    
    print("\n" + "=" * 80)
    print("SCENARIO SETUP")
    print("=" * 80)
    print(f"Player: {player_species}")
    print(f"Opponent: {opponent_species}")
    print("\nGoal: Verify that stat drops from Draco Meteor/Leaf Storm are tracked")
    print("Expected: SpA drops by -2 after each use")
    
    # Create Pokemon
    player_mon, player_role, player_moves, player_ability, player_item = create_pokemon_from_randbats(
        player_species, randbats_data[player_species], gen_data
    )
    
    opponent_mon, opp_role, opp_moves, opp_ability, opp_item = create_pokemon_from_randbats(
        opponent_species, randbats_data[opponent_species], gen_data
    )
    
    # Set identifiers
    player_mon._identifier_string = f"p1: {player_species}"
    opponent_mon._identifier_string = f"p2: {opponent_species}"
    
    # Display Pokemon details
    print("\n" + "=" * 80)
    print("POKEMON DETAILS")
    print("=" * 80)
    
    print(f"\n{player_species} ({player_role}):")
    print(f"  Level: {player_mon.level}")
    print(f"  HP: {player_mon.stats['hp']}")
    print(f"  SpA: {player_mon.stats['spa']}")
    print(f"  Moves: {', '.join(player_moves)}")
    
    print(f"\n{opponent_species} ({opp_role}):")
    print(f"  Level: {opponent_mon.level}")
    print(f"  HP: {opponent_mon.stats['hp']}")
    print(f"  SpD: {opponent_mon.stats['spd']}")
    
    # Check for stat-dropping moves
    print("\n" + "=" * 80)
    print("CHECKING MOVE PROPERTIES")
    print("=" * 80)
    
    from bot.mcts.shadow_state import get_move_boosts
    
    for move_name, move in player_mon.moves.items():
        print(f"\n{move_name}:")
        print(f"  Base Power: {getattr(move, 'base_power', 0)}")
        print(f"  Category: {getattr(move, 'category', 'unknown')}")
        
        # Check for self-inflicted stat changes
        self_data = getattr(move, 'self', None)
        if self_data:
            print(f"  move.self: {self_data}")
        
        boosts_data = getattr(move, 'boosts', None)
        if boosts_data:
            print(f"  move.boosts: {boosts_data}")
        
        # Use our helper function
        boost_result = get_move_boosts(move)
        if boost_result:
            self_boosts, target_boosts, chance = boost_result
            print(f"  get_move_boosts result:")
            print(f"    self_boosts: {self_boosts}")
            print(f"    target_boosts: {target_boosts}")
            print(f"    chance: {chance * 100:.0f}%")
            
            if self_boosts and any(v < 0 for v in self_boosts.values()):
                print(f"    ⚠️  WARNING: This move drops our stats!")
    
    # Test direct state tracking
    print("\n" + "=" * 80)
    print("TESTING STAT DROP TRACKING")
    print("=" * 80)
    
    from bot.mcts.shadow_state import ShadowState
    
    # Create battle
    team = {f"p1: {player_species}": player_mon}
    opp_team = {f"p2: {opponent_species}": opponent_mon}
    
    battle = create_mock_battle(
        active_identifier=f"p1: {player_species}",
        active=player_mon,
        opponent_identifier=f"p2: {opponent_species}",
        opponent=opponent_mon,
        team=team,
        opponent_team=opp_team
    )
    
    battle.available_moves = list(player_mon.moves.values())
    
    ctx_me = EvalContext(me=player_mon, opp=opponent_mon, battle=battle, cache={})
    ctx_opp = EvalContext(me=opponent_mon, opp=player_mon, battle=battle, cache={})
    
    # Create initial state
    state = ShadowState.from_battle(
        battle=battle,
        ctx_me=ctx_me,
        ctx_opp=ctx_opp,
        score_move_fn=score_move,
        score_switch_fn=score_switch,
        dmg_fn=estimate_damage_fraction,
        opp_tau=8.0,
        status_threshold=0.30,
        model_miss=False,  # Disable misses for clearer testing
        model_crit=False,  # Disable crits for clearer testing
        debug=True,
    )
    
    print(f"\nInitial state:")
    print(f"  Our boosts: {state.my_boosts.get(id(player_mon), {})}")
    print(f"  Opponent boosts: {state.opp_boosts.get(id(opponent_mon), {})}")
    
    # Find Draco Meteor or Leaf Storm
    draco_meteor = player_mon.moves.get('dracometeor')
    leaf_storm = player_mon.moves.get('leafstorm')
    
    test_move = draco_meteor or leaf_storm
    test_move_name = 'dracometeor' if draco_meteor else 'leafstorm'
    
    if not test_move:
        print("\n❌ ERROR: Hydrapple doesn't have Draco Meteor or Leaf Storm!")
        print(f"Available moves: {list(player_mon.moves.keys())}")
        return
    
    print(f"\nTesting with: {test_move_name}")
    
    # Simulate using the move 3 times
    rng = random.Random(42)
    
    for i in range(1, 4):
        print(f"\n--- Use #{i}: {test_move_name} ---")
        
        # Apply the move
        next_state = state.step(("move", test_move), rng=rng)
        
        # Check boosts
        our_boosts = next_state.my_boosts.get(id(player_mon), {})
        print(f"  Our boosts after use: {our_boosts}")
        print(f"  SpA boost: {our_boosts.get('spa', 0):+d}")
        
        if our_boosts.get('spa', 0) == -2 * i:
            print(f"  ✅ Correct! SpA dropped to {-2 * i}")
        elif our_boosts.get('spa', 0) == -6 and i >= 3:
            print(f"  ✅ Correct! SpA clamped at -6")
        else:
            print(f"  ❌ WRONG! Expected SpA={-2*i}, got {our_boosts.get('spa', 0)}")
        
        # Check events
        if hasattr(next_state, 'events'):
            print(f"  Events: {next_state.events}")
        
        state = next_state
    
    # Run MCTS to see if it understands the stat drops
    print("\n" + "=" * 80)
    print("RUNNING MCTS")
    print("=" * 80)
    print("Testing if MCTS values moves differently after stat drops")
    
    cfg = MCTSConfig(
        num_simulations=200,
        max_depth=3,
        c_puct=1.6,
        seed=42,
        use_hybrid_expansion=False,
    )
    
    # Reset to initial state
    state_fresh = ShadowState.from_battle(
        battle=battle,
        ctx_me=ctx_me,
        ctx_opp=ctx_opp,
        score_move_fn=score_move,
        score_switch_fn=score_switch,
        dmg_fn=estimate_damage_fraction,
        opp_tau=8.0,
        status_threshold=0.30,
        model_miss=False,
        model_crit=False,
        debug=True,
    )
    
    try:
        picked, stats = search(
            battle=battle,
            ctx_me=ctx_me,
            ctx_opp=ctx_opp,
            score_move_fn=score_move,
            score_switch_fn=score_switch,
            dmg_fn=estimate_damage_fraction,
            cfg=cfg,
            opp_tau=8.0,
            return_stats=True,
            return_tree=True,
        )
        
        print("\nMCTS Results:")
        print(f"  Total simulations: {stats['root'].N}")
        
        # Show visit distribution
        print("\n  Visit Distribution:")
        for action, child in sorted(stats['root'].children.items(), key=lambda x: x[1].N, reverse=True):
            kind, obj = action
            name = getattr(obj, 'id', 'unknown')
            visits = child.N
            q_val = child.Q
            
            print(f"    {name:20s}: {visits:4d} visits ({visits/stats['root'].N*100:5.1f}%), Q={q_val:+.3f}")
        
        # Analyze Draco Meteor/Leaf Storm path
        print(f"\n  Analyzing {test_move_name} path:")
        
        for action, child in stats['root'].children.items():
            if action[0] == 'move':
                move_name = getattr(action[1], 'id', '')
                if test_move_name in move_name.lower():
                    print(f"\n  After using {test_move_name}:")
                    print(f"    Visits: {child.N}")
                    print(f"    Q-value: {child.Q:+.3f}")
                    
                    # Check state boosts
                    our_boosts = child.state.my_boosts.get(id(player_mon), {})
                    print(f"    Our boosts in state: {our_boosts}")
                    print(f"    SpA: {our_boosts.get('spa', 0):+d}")
                    
                    if our_boosts.get('spa', 0) == -2:
                        print(f"    ✅ State correctly shows -2 SpA!")
                    else:
                        print(f"    ❌ State doesn't show -2 SpA drop!")
                    
                    # Check if MCTS explored what happens after
                    if child.children:
                        print(f"\n    Opponent responses explored: {len(child.children)}")
                        
                        # Show top response
                        top_opp = max(child.children.items(), key=lambda x: x[1].N)
                        opp_action, opp_child = top_opp
                        opp_move = getattr(opp_action[1], 'id', 'unknown')
                        
                        print(f"    Top opponent response: {opp_move} ({opp_child.N} visits)")
                        
                        # Check what happens if we use the move again
                        if opp_child.children:
                            print(f"\n    Our follow-up moves:")
                            for our_action, our_child in sorted(opp_child.children.items(), key=lambda x: x[1].N, reverse=True)[:3]:
                                our_move = getattr(our_action[1], 'id', 'unknown')
                                our_boosts_after = our_child.state.my_boosts.get(id(player_mon), {})
                                
                                print(f"      {our_move}: {our_child.N} visits, Q={our_child.Q:+.3f}")
                                print(f"        SpA boost: {our_boosts_after.get('spa', 0):+d}")
                                
                                # If we use Draco/Leaf again, should be -4
                                if test_move_name in our_move.lower() and our_boosts_after.get('spa', 0) == -4:
                                    print(f"        ✅ Correctly shows -4 SpA after second use!")
        
        # Visualize tree
        print("\n" + "=" * 80)
        print("TREE VISUALIZATION")
        print("=" * 80)
        visualize_tree_depth(stats['root'], max_depth=2)
        
    except Exception as e:
        print(f"\nMCTS Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_hydrapple_stat_drops()