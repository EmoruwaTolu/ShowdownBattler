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
    get_move_boosts,
    get_move_self,
    create_pokemon_from_randbats,
)


def test_boosting_moves():
    """
    Test scenario: Pokemon with stat-boosting moves
    
    Tests:
    1. Heuristic scoring of setup moves (Swords Dance, Nasty Plot, Dragon Dance)
    2. MCTS understanding of when to setup vs attack
    3. How boost evaluation affects state values
    
    Scenarios to test:
    - Safe setup: Can survive opponent's attacks
    - Risky setup: Opponent can OHKO/2HKO us
    - Already boosted: Should attack instead of setup again
    """
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "STAT BOOST MOVE TEST" + " " * 37 + "║")
    print("╚" + "=" * 78 + "╝")
    
    gen_data = GenData.from_gen(9)
    randbats_data = load_randbats_data()
    
    # Test multiple scenarios
    scenarios = [
        {
            "name": "Safe Setup - Garchomp vs Clodsire",
            "player": "Garchomp",  # Has Swords Dance
            "opponent": "Clodsire",  # Passive, won't OHKO
            "description": "Garchomp can safely setup Swords Dance against passive Clodsire"
        },
        {
            "name": "Nasty Plot Sweep - Hatterene vs Tyranitar", 
            "player": "Hatterene",  # Has Nasty Plot
            "opponent": "Tyranitar",
            "description": "Special attacker boosting SpA"
        },
        {
            "name": "Dragon Dance - Dragonite vs Gastrodon",
            "player": "Dragonite",  # Has Dragon Dance
            "opponent": "Gastrodon",
            "description": "Boost Attack + Speed simultaneously"
        },
    ]
    
    for scenario in scenarios:
        run_boosting_scenario(scenario, gen_data, randbats_data)


def run_boosting_scenario(scenario: dict, gen_data, randbats_data):
    """Run a single boosting move test scenario"""
    
    print("\n" + "=" * 80)
    print(f"SCENARIO: {scenario['name']}")
    print("=" * 80)
    print(f"Description: {scenario['description']}")
    
    player_species = scenario['player']
    opponent_species = scenario['opponent']
    
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
    print("\n" + "-" * 80)
    print("POKEMON DETAILS")
    print("-" * 80)
    
    print(f"\n{player_species} ({player_role}):")
    print(f"  Level: {player_mon.level}")
    print(f"  HP: {player_mon.stats['hp']}")
    print(f"  Atk: {player_mon.stats['atk']}, SpA: {player_mon.stats['spa']}")
    print(f"  Moves: {', '.join(player_moves)}")
    
    print(f"\n{opponent_species} ({opp_role}):")
    print(f"  Level: {opponent_mon.level}")
    print(f"  HP: {opponent_mon.stats['hp']}")
    print(f"  Def: {opponent_mon.stats['def']}, SpD: {opponent_mon.stats['spd']}")
    
    # Check for boosting moves
    print("\n" + "-" * 80)
    print("MOVE ANALYSIS")
    print("-" * 80)
    
    from bot.mcts.shadow_state import get_move_boosts
    
    boosting_moves = []
    attacking_moves = []
    
    for move_name, move in player_mon.moves.items():
        bp = getattr(move, 'base_power', 0)
        cat = getattr(move, 'category', 'unknown')
        
        print(f"\n{move_name}:")
        print(f"  Base Power: {bp}")
        print(f"  Category: {cat}")
        
        boost_result = get_move_boosts(move)
        if boost_result:
            self_boosts, target_boosts, chance = boost_result
            
            if self_boosts:
                boosting_moves.append((move_name, move, self_boosts))
                print(f"  Self-boosts: {self_boosts}")
                print(f"  ✨ SETUP MOVE!")
            
            if target_boosts:
                print(f"  Target-boosts: {target_boosts} ({chance*100:.0f}% chance)")
        
        if bp > 0:
            attacking_moves.append((move_name, move, bp))
    
    if not boosting_moves:
        print(f"\n⚠️  {player_species} has no boosting moves! Skipping scenario.")
        return
    
    # Heuristic scoring
    print("\n" + "-" * 80)
    print("HEURISTIC SCORES (before MCTS)")
    print("-" * 80)
    
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
    
    ctx_me = EvalContext(me=player_mon, opp=opponent_mon, battle=battle, cache={})
    
    # Score all moves
    move_scores = []
    for move_name, move in player_mon.moves.items():
        score = score_move(move, battle, ctx_me)
        move_scores.append((move_name, score, move))
    
    # Sort by score
    move_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\nAll moves ranked:")
    for move_name, score, move in move_scores:
        bp = getattr(move, 'base_power', 0)
        
        # Check if boosting
        boost_result = get_move_boosts(move)
        boost_str = ""
        if boost_result:
            self_boosts, target_boosts, chance = boost_result
            if self_boosts:
                boost_str = f" [BOOST: {self_boosts}]"
        
        print(f"  {move_name:20s}: score={score:6.2f}, BP={bp:3d}{boost_str}")
    
    # Analyze risk
    print("\n" + "-" * 80)
    print("SETUP RISK ANALYSIS")
    print("-" * 80)
    
    # Check if opponent can OHKO/2HKO us
    max_opp_damage = 0.0
    strongest_move = None
    
    for opp_move in opponent_mon.moves.values():
        dmg = estimate_damage_fraction(opp_move, opponent_mon, player_mon, battle)
        if dmg > max_opp_damage:
            max_opp_damage = dmg
            strongest_move = getattr(opp_move, 'id', 'unknown')
    
    print(f"\nOpponent's strongest move: {strongest_move}")
    print(f"  Damage to us: {max_opp_damage:.1%}")
    
    if max_opp_damage >= 1.0:
        print(f"  ⚠️  DANGER: Opponent can OHKO us! Setup is very risky.")
    elif max_opp_damage >= 0.5:
        print(f"  ⚠️  WARNING: Opponent can 2HKO us. Setup is risky.")
    else:
        print(f"  ✅ SAFE: We can survive multiple hits. Setup is viable!")
    
    # Run MCTS
    print("\n" + "-" * 80)
    print("RUNNING MCTS")
    print("-" * 80)
    
    cfg = MCTSConfig(
        num_simulations=400,
        max_depth=4,
        c_puct=1.6,
        seed=42,
        use_hybrid_expansion=False,
    )
    
    battle.available_moves = list(player_mon.moves.values())
    
    from bot.mcts.shadow_state import ShadowState
    
    state = ShadowState.from_battle(
        battle=battle,
        ctx_me=ctx_me,
        ctx_opp=EvalContext(me=opponent_mon, opp=player_mon, battle=battle, cache={}),
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
            ctx_opp=EvalContext(me=opponent_mon, opp=player_mon, battle=battle, cache={}),
            score_move_fn=score_move,
            score_switch_fn=score_switch,
            dmg_fn=estimate_damage_fraction,
            cfg=cfg,
            opp_tau=8.0,
            return_stats=True,
            return_tree=True,
        )
        
        print("\nMCTS Results:")
        print(f"  Simulations: {stats['root'].N}")
        print(f"  Picked: {picked[0]} {getattr(picked[1], 'id', 'unknown')}")
        
        # Visit distribution
        print("\n  Visit Distribution:")
        for action, child in sorted(stats['root'].children.items(), 
                                   key=lambda x: x[1].N, reverse=True):
            kind, obj = action
            name = getattr(obj, 'id', 'unknown')
            visits = child.N
            q_val = child.Q
            
            # Check if boosting move
            boost_marker = ""
            if kind == 'move':
                boost_result = get_move_boosts(obj)
                if boost_result:
                    self_boosts, _, _ = boost_result
                    if self_boosts:
                        boost_marker = " ✨"
            
            print(f"    {name:20s}: {visits:4d} visits ({visits/stats['root'].N*100:5.1f}%), "
                  f"Q={q_val:+.3f}{boost_marker}")
        
        # Analyze boosting move paths
        print("\n" + "-" * 80)
        print("BOOSTING MOVE ANALYSIS")
        print("-" * 80)
        
        for move_name, move, boosts in boosting_moves:
            move_id = getattr(move, 'id', move_name)
            
            for action, child in stats['root'].children.items():
                if action[0] == 'move' and getattr(action[1], 'id', '') == move_id:
                    print(f"\n{move_name} (boosts: {boosts}):")
                    print(f"  Visits: {child.N} ({child.N/stats['root'].N*100:.1f}%)")
                    print(f"  Q-value: {child.Q:+.3f}")
                    
                    # Check state after boost
                    our_boosts = child.state.my_boosts.get(id(player_mon), {})
                    print(f"  Our boosts after: {our_boosts}")
                    
                    # Check what MCTS does after boosting
                    if child.children:
                        print(f"\n  After boosting, opponent responds:")
                        
                        # Find most visited opponent response
                        top_opp = max(child.children.items(), key=lambda x: x[1].N)
                        opp_action, opp_child = top_opp
                        opp_move = getattr(opp_action[1], 'id', 'unknown')
                        
                        print(f"    Top response: {opp_move} ({opp_child.N} visits)")
                        
                        # Then what do we do?
                        if opp_child.children:
                            print(f"\n  Then we follow up with:")
                            
                            follow_ups = sorted(opp_child.children.items(), 
                                              key=lambda x: x[1].N, reverse=True)[:3]
                            
                            for our_action, our_child in follow_ups:
                                our_move = getattr(our_action[1], 'id', 'unknown')
                                our_boosts_after = our_child.state.my_boosts.get(id(player_mon), {})
                                
                                # Check if we boost again or attack
                                is_boost = False
                                for bm_name, bm_move, _ in boosting_moves:
                                    if getattr(bm_move, 'id', '') == our_move:
                                        is_boost = True
                                        break
                                
                                marker = "✨ BOOST AGAIN" if is_boost else "⚔️  ATTACK"
                                
                                print(f"    {our_move:20s}: {our_child.N:3d} visits, "
                                      f"Q={our_child.Q:+.3f} {marker}")
                                print(f"      Boosts: {our_boosts_after}")
        
        # Tree visualization
        print("\n" + "-" * 80)
        print("TREE VISUALIZATION")
        print("-" * 80)
        visualize_tree_depth(stats['root'], max_depth=3)
        
        # Interpretation
        print("\n" + "-" * 80)
        print("INTERPRETATION")
        print("-" * 80)
        
        # Check what MCTS preferred
        top_action = max(stats['root'].children.items(), key=lambda x: x[1].N)
        top_name = getattr(top_action[0][1], 'id', 'unknown')
        
        is_setup = False
        for bm_name, bm_move, _ in boosting_moves:
            if getattr(bm_move, 'id', '') == top_name:
                is_setup = True
                break
        
        if is_setup:
            print(f"\n✨ MCTS chose to SETUP with {top_name}!")
            print(f"   This suggests:")
            print(f"   - Setup is safe enough")
            print(f"   - The boost value outweighs immediate damage")
            print(f"   - MCTS expects to win after boosting")
        else:
            print(f"\n⚔️  MCTS chose to ATTACK with {top_name}!")
            print(f"   This suggests:")
            print(f"   - Setup is too risky OR")
            print(f"   - Immediate KO is better OR")
            print(f"   - Boost doesn't provide enough value")
        
    except Exception as e:
        print(f"\n❌ MCTS Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_boosting_moves()