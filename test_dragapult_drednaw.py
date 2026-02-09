import sys
import os
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List
from unittest.mock import Mock

from poke_env.battle import MoveCategory, PokemonType
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
    create_pokemon_from_randbats,
)


def test_dragapult_vs_drednaw():
    """
    Custom scenario: Dragapult vs Drednaw
    
    This is an interesting matchup because:
    - Dragapult is fast, offensive Ghost/Dragon
    - Drednaw is slow, bulky Water/Rock
    - Type advantage favors Dragapult (Dragon moves vs Water/Rock)
    """
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 24 + "CUSTOM SCENARIO" + " " * 39 + "║")
    print("║" + " " * 20 + "DRAGAPULT VS DREDNAW" + " " * 37 + "║")
    print("╚" + "=" * 78 + "╝")
    
    gen_data = GenData.from_gen(9)
    randbats_data = load_randbats_data()
    
    # Get the specific Pokemon data
    player_species = "Dragapult"
    opponent_species = "Drednaw"
    
    print("\n" + "=" * 80)
    print("MATCHUP SETUP")
    print("=" * 80)
    print(f"Player: {player_species}")
    print(f"Opponent: {opponent_species}")
    
    # Check if Pokemon exist in data
    if player_species not in randbats_data:
        print(f"ERROR: {player_species} not found in RandBats data!")
        return
    
    if opponent_species not in randbats_data:
        print(f"ERROR: {opponent_species} not found in RandBats data!")
        return
    
    # Show available roles
    print(f"\n{player_species} roles: {list(randbats_data[player_species]['roles'].keys())}")
    print(f"{opponent_species} roles: {list(randbats_data[opponent_species]['roles'].keys())}")
    
    # Create Pokemon (use first role by default)
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
    print(f"  Types: {', '.join(str(t).split('.')[-1] for t in player_mon.types)}")
    print(f"  HP: {player_mon.stats['hp']}")
    print(f"  Atk: {player_mon.stats['atk']}")
    print(f"  Def: {player_mon.stats['def']}")
    print(f"  SpA: {player_mon.stats['spa']}")
    print(f"  SpD: {player_mon.stats['spd']}")
    print(f"  Spe: {player_mon.stats['spe']}")
    print(f"  Ability: {player_ability} (not used in heuristics)")
    print(f"  Item: {player_item} (not used in heuristics)")
    print(f"  Moves: {', '.join(player_moves)}")
    
    print(f"\n{opponent_species} ({opp_role}):")
    print(f"  Level: {opponent_mon.level}")
    print(f"  Types: {', '.join(str(t).split('.')[-1] for t in opponent_mon.types)}")
    print(f"  HP: {opponent_mon.stats['hp']}")
    print(f"  Atk: {opponent_mon.stats['atk']}")
    print(f"  Def: {opponent_mon.stats['def']}")
    print(f"  SpA: {opponent_mon.stats['spa']}")
    print(f"  SpD: {opponent_mon.stats['spd']}")
    print(f"  Spe: {opponent_mon.stats['spe']}")
    print(f"  Ability: {opp_ability} (not used in heuristics)")
    print(f"  Item: {opp_item} (not used in heuristics)")
    print(f"  Moves: {', '.join(opp_moves)}")
    
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
    
    # Create contexts
    ctx_me = EvalContext(me=player_mon, opp=opponent_mon, battle=battle, cache={})
    ctx_opp = EvalContext(me=opponent_mon, opp=player_mon, battle=battle, cache={})
    
    # Show heuristic scores
    print("\n" + "=" * 80)
    print("HEURISTIC SCORES (before MCTS)")
    print("=" * 80)
    
    print(f"\n{player_species}'s moves:")
    for move_id, move in sorted(player_mon.moves.items()):
        try:
            score = score_move(move, battle, ctx_me)
            dmg = estimate_damage_fraction(move, player_mon, opponent_mon, battle)
            
            # Get move details
            move_power = getattr(move, 'base_power', 0)
            move_type = str(getattr(move, 'type', 'NORMAL')).split('.')[-1]
            move_cat = str(getattr(move, 'category', 'SPECIAL')).split('.')[-1]
            move_acc = getattr(move, 'accuracy', 1.0)
            
            print(f"  {move_id:20s}: score={score:6.2f}, dmg={dmg:.3f}")
            print(f"    {move_type:8s} {move_cat:8s} | Power: {move_power:3d} | Acc: {move_acc:.2f}")
        except Exception as e:
            print(f"  {move_id:20s}: ERROR - {e}")
    
    # Run MCTS
    print("\n" + "=" * 80)
    print("RUNNING MCTS")
    print("=" * 80)
    print("Simulations: 400")
    print("Max Depth: 4")
    print("c_puct: 1.6")
    print("opp_tau: 8.0 (default)")
    
    cfg = MCTSConfig(
        num_simulations=400,
        max_depth=4,
        c_puct=1.6,
        seed=42,
        use_hybrid_expansion=False,
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
            opp_tau=8.0,  # Default
            return_stats=True,
            return_tree=True,
        )
        
        # Show MCTS results
        print("\n" + "=" * 80)
        print("MCTS RESULTS")
        print("=" * 80)
        
        kind, obj = picked
        picked_name = getattr(obj, 'id', getattr(obj, 'species', 'unknown'))
        print(f"\nMCTS Picked: {kind} {picked_name}")
        
        print("\nTop 5 Actions by Visits:")
        for i, action_stats in enumerate(stats['top'][:5], 1):
            print(f"  {i}. {action_stats['kind']} {action_stats['name']}")
            print(f"     Visits: {action_stats['visits']:4d} ({action_stats['visits']/400*100:5.1f}%), Q: {action_stats['q']:+.3f}, Prior: {action_stats['prior']:.4f}")
        
        # Visualize tree
        root = stats['root']
        print("\n" + "=" * 80)
        print("TREE VISUALIZATION")
        print("=" * 80)
        visualize_tree_depth(root, max_depth=2)
        
        # Analysis
        print("\n" + "=" * 80)
        print("DETAILED ANALYSIS")
        print("=" * 80)
        print(f"Total simulations: {root.N}")
        print(f"Root Q-value: {root.Q:+.3f}")
        
        if root.Q > 0.5:
            print(f"  → Dragapult is winning! (Q > 0.5)")
        elif root.Q > 0.0:
            print(f"  → Dragapult has slight advantage (Q > 0)")
        elif root.Q > -0.5:
            print(f"  → Close matchup, slight disadvantage (Q ∈ (-0.5, 0))")
        else:
            print(f"  → Dragapult is losing (Q < -0.5)")
        
        print(f"Number of root actions: {len(root.children)}")
        
        # Visit distribution
        print("\nVisit Distribution:")
        total_visits = sum(child.N for child in root.children.values())
        for action, child in sorted(root.children.items(), key=lambda x: x[1].N, reverse=True):
            kind = action[0]
            obj = action[1]
            name = getattr(obj, 'id', getattr(obj, 'species', 'unknown'))
            pct = (child.N / total_visits * 100) if total_visits > 0 else 0
            print(f"  {name:25s}: {child.N:4d} visits ({pct:5.1f}%), Q={child.Q:+.3f}")
        
        # Show depth 1 children for top move
        print("\n" + "=" * 80)
        print(f"WHAT HAPPENS AFTER {picked_name.upper()}?")
        print("=" * 80)
        
        picked_node = root.children[picked]
        if picked_node.children:
            print(f"\nAfter {picked_name}, opponent's likely responses:")
            opp_children = sorted(
                picked_node.children.items(),
                key=lambda x: x[1].N,
                reverse=True
            )[:5]
            
            for i, (opp_action, opp_child) in enumerate(opp_children, 1):
                opp_kind = opp_action[0]
                opp_obj = opp_action[1]
                opp_name = getattr(opp_obj, 'id', getattr(opp_obj, 'species', 'unknown'))
                opp_pct = (opp_child.N / picked_node.N * 100) if picked_node.N > 0 else 0
                
                print(f"\n  {i}. Opponent uses {opp_obj}")
                print(f"     Explored: {opp_child.N:4d} times ({opp_pct:5.1f}% of simulations)")
                print(f"     Q-value: {opp_child.Q:+.3f}")
                print(f"     State: {opp_child.state.my_active.species} {opp_child.state.my_active_hp():.2f} HP vs {opp_child.state.opp_active.species} {opp_child.state.opp_active_hp():.2f} HP")
        else:
            print(f"\nNo children (probably terminal state after {picked_name})")
        
    except Exception as e:
        print(f"\nMCTS Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_dragapult_vs_drednaw()
    
    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)