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
    create_pokemon_from_randbats,
)


def test_entei_vs_tyranitar():
    """
    Test scenario: Entei vs Tyranitar
    
    This tests secondary effect scoring because:
    - Entei has Sacred Fire (50% burn chance)
    - Entei has Extreme Speed (priority move)
    - Entei has Stomping Tantrum (gets stronger after failed move)
    - Tyranitar is a physical attacker (Rock/Dark) that's weak to Fighting
    
    Key questions:
    1. Does MCTS value Sacred Fire's 50% burn chance?
    2. Does it recognize burning a physical attacker is valuable?
    3. How does it compare Sacred Fire vs Extreme Speed?
    """
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 24 + "CUSTOM SCENARIO" + " " * 39 + "║")
    print("║" + " " * 22 + "ENTEI VS TYRANITAR" + " " * 39 + "║")
    print("╚" + "=" * 78 + "╝")
    
    gen_data = GenData.from_gen(9)
    randbats_data = load_randbats_data()
    
    # Get the specific Pokemon data
    player_species = "Entei"
    opponent_species = "Tyranitar"
    
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
    
    # Create Pokemon (use first role by default, or random)
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
    print(f"  Ability: {player_ability}")
    print(f"  Item: {player_item}")
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
    print(f"  Ability: {opp_ability}")
    print(f"  Item: {opp_item}")
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

    print("\n" + "=" * 80)
    print("SACRED FIRE DIAGNOSTIC")
    print("=" * 80)

    # Get Sacred Fire from the player's moves
    sacred_fire = player_mon.moves.get('sacredfire')

    if not sacred_fire:
        print("❌ ERROR: Sacred Fire not in player_mon.moves!")
        print(f"Available moves: {list(player_mon.moves.keys())}")
    else:
        print(f"✅ Sacred Fire found in moves")
        print(f"\nMove attributes:")
        print(f"  id: {getattr(sacred_fire, 'id', 'NOT SET')}")
        print(f"  base_power: {getattr(sacred_fire, 'base_power', 'NOT SET')}")
        print(f"  accuracy: {getattr(sacred_fire, 'accuracy', 'NOT SET')}")
        print(f"  status: {getattr(sacred_fire, 'status', 'NOT SET')}")
        print(f"  secondary: {getattr(sacred_fire, 'secondary', 'NOT SET')}")
        
        # Check secondary in detail
        secondary = getattr(sacred_fire, 'secondary', None)
        print(f"\nSecondary detailed check:")
        print(f"  Type: {type(secondary)}")
        print(f"  Value: {secondary}")
        
        if secondary:
            if isinstance(secondary, list):
                print(f"  Is list: True")
                print(f"  Length: {len(secondary)}")
                for i, item in enumerate(secondary):
                    print(f"  [{i}]: {item}")
                    print(f"       Type: {type(item)}")
                    if isinstance(item, dict):
                        print(f"       Keys: {list(item.keys())}")
                        print(f"       chance: {item.get('chance', 'NOT SET')}")
                        print(f"       status: {item.get('status', 'NOT SET')}")
            else:
                print(f"  ❌ Secondary is not a list! Type: {type(secondary)}")
        else:
            print(f"  ❌ Secondary is None or empty!")
        
        # Test status_infliction directly
        print(f"\nTesting status_infliction():")
        from bot.mcts.shadow_state import status_infliction
        
        result = status_infliction(sacred_fire)
        print(f"  Result: {result}")
        
        if result:
            status, prob = result
            print(f"  ✅ Status: {status}")
            print(f"  ✅ Probability: {prob} ({prob*100:.0f}%)")
        else:
            print(f"  ❌ status_infliction returned None!")
            
        # Also check what GenData has for Sacred Fire
        print(f"\nChecking GenData for Sacred Fire:")
        from test_random_randbats import get_move_info, get_move_secondary
        
        move_info = get_move_info("Sacred Fire", gen_data)
        if move_info:
            print(f"  GenData has Sacred Fire: True")
            print(f"  secondary in GenData: {move_info.get('secondary', 'NOT SET')}")
            
            secondary_from_gen = get_move_secondary("Sacred Fire", gen_data)
            print(f"  get_move_secondary result: {secondary_from_gen}")
        else:
            print(f"  ❌ GenData doesn't have Sacred Fire!")

    print("=" * 80)
    
    # Show heuristic scores with detailed breakdown
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
            move_status = getattr(move, 'status', None)
            move_secondary = getattr(move, 'secondary', None)
            
            print(f"\n  {move_id:20s}: score={score:6.2f}, dmg={dmg:.3f}")
            print(f"    Type: {move_type:8s} | Cat: {move_cat:8s} | Power: {move_power:3d} | Acc: {move_acc:.2f}")
            
            # Show status if present
            if move_status:
                print(f"    Status: {move_status} (guaranteed)")
            
            # Show secondary if present
            if move_secondary:
                if isinstance(move_secondary, list):
                    for sec in move_secondary:
                        if isinstance(sec, dict):
                            chance = sec.get('chance', 100)
                            status = sec.get('status', None)
                            boosts = sec.get('boosts', None)
                            volatileStatus = sec.get('volatileStatus', None)
                            
                            if status:
                                print(f"    Secondary: {chance}% chance to inflict {status}")
                            if boosts:
                                print(f"    Secondary: {chance}% chance to boost {boosts}")
                            if volatileStatus:
                                print(f"    Secondary: {chance}% chance to inflict {volatileStatus}")
                else:
                    print(f"    Secondary: {move_secondary}")
            
        except Exception as e:
            print(f"  {move_id:20s}: ERROR - {e}")
            import traceback
            traceback.print_exc()
    
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
            opp_tau=8.0,
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
        
        print("\nTop Actions by Visits:")
        for i, action_stats in enumerate(stats['top'][:10], 1):
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
        print("SECONDARY EFFECTS ANALYSIS")
        print("=" * 80)
        
        # Check if Sacred Fire (with 50% burn) is being valued
        sacred_fire_stats = None
        for action_stats in stats['top']:
            if 'sacredfire' in action_stats['name'].lower():
                sacred_fire_stats = action_stats
                break
        
        if sacred_fire_stats:
            print(f"\nSacred Fire Analysis:")
            print(f"  Visits: {sacred_fire_stats['visits']} ({sacred_fire_stats['visits']/400*100:.1f}%)")
            print(f"  Q-value: {sacred_fire_stats['q']:+.3f}")
            print(f"  Prior: {sacred_fire_stats['prior']:.4f}")
            
            # Check if it's exploring the burn path
            sacred_fire_action = None
            for action, child in root.children.items():
                if action[0] == 'move' and hasattr(action[1], 'id'):
                    if 'sacredfire' in action[1].id.lower():
                        sacred_fire_action = action
                        break
            
            if sacred_fire_action:
                sacred_fire_node = root.children[sacred_fire_action]
                print(f"\n  After Sacred Fire (exploring secondary effects):")
                
                # Look at the different paths
                if sacred_fire_node.children:
                    print(f"    Number of opponent responses explored: {len(sacred_fire_node.children)}")
                    
                    # Check if we're seeing different outcomes (burn vs no burn)
                    print(f"\n    Top opponent responses:")
                    for i, (opp_action, opp_child) in enumerate(
                        sorted(sacred_fire_node.children.items(), key=lambda x: x[1].N, reverse=True)[:5], 
                        1
                    ):
                        opp_name = getattr(opp_action[1], 'id', 'unknown')
                        print(f"      {i}. {opp_name}: {opp_child.N} visits, Q={opp_child.Q:+.3f}")
                        
                        # Check opponent's status
                        opp_status = opp_child.state.opp_active_status()
                        if opp_status:
                            print(f"         → Opponent status: {opp_status}")
                        else:
                            print(f"         → Opponent not burned")
        
        print("\n" + "=" * 80)
        print("DETAILED ANALYSIS")
        print("=" * 80)
        print(f"Total simulations: {root.N}")
        print(f"Root Q-value: {root.Q:+.3f}")
        
        if root.Q > 0.5:
            print(f"  → Entei is winning! (Q > 0.5)")
        elif root.Q > 0.0:
            print(f"  → Entei has slight advantage (Q > 0)")
        elif root.Q > -0.5:
            print(f"  → Close matchup, slight disadvantage (Q ∈ (-0.5, 0))")
        else:
            print(f"  → Entei is losing (Q < -0.5)")
        
        print(f"Number of root actions: {len(root.children)}")

        def test_sacred_fire_burn_proc():
            """Direct test: Does Sacred Fire actually apply burns?"""
            print("\n" + "=" * 80)
            print("DIRECT BURN PROC TEST")
            print("=" * 80)
            
            from bot.mcts.shadow_state import ShadowState, status_infliction
            from poke_env.battle import Status
            import random
            
            # Check status_infliction
            sacred_fire = player_mon.moves.get('sacredfire')
            result = status_infliction(sacred_fire)
            
            print(f"\n1. status_infliction(sacred_fire): {result}")
            
            if not result:
                print("   ❌ BROKEN: Returns None!")
                return 0
            
            status, prob = result
            print(f"   Status: {status}, Probability: {prob}")
            
            # Test ShadowState
            state = ShadowState.from_battle(
                battle=battle,
                ctx_me=ctx_me,
                ctx_opp=ctx_opp,
                score_move_fn=score_move,
                score_switch_fn=score_switch,
                dmg_fn=estimate_damage_fraction,
                opp_tau=8.0,
                status_threshold=0.25,
                model_miss=True,
                model_crit=True,
            )
            
            # Test 100 times
            burn_count = 0
            for i in range(100):
                rng = random.Random(i)
                next_state = state.step(("move", sacred_fire), rng=rng)
                
                if next_state.opp_active_status() == Status.BRN:
                    burn_count += 1
            
            print(f"\n2. Burns in 100 simulations: {burn_count}/100")
            print(f"   Expected: ~50")
            
            if burn_count == 0:
                print("   ❌ NO BURNS PROCCING!")
            elif 40 <= burn_count <= 60:
                print("   ✅ WORKING!")
            else:
                print(f"   ⚠️  Unusual rate")
            
            return burn_count
        
        test_sacred_fire_burn_proc()
        
        # Visit distribution
        print("\nVisit Distribution:")
        total_visits = sum(child.N for child in root.children.values())
        for action, child in sorted(root.children.items(), key=lambda x: x[1].N, reverse=True):
            kind = action[0]
            obj = action[1]
            name = getattr(obj, 'id', getattr(obj, 'species', 'unknown'))
            pct = (child.N / total_visits * 100) if total_visits > 0 else 0
            print(f"  {name:25s}: {child.N:4d} visits ({pct:5.1f}%), Q={child.Q:+.3f}")
        
    except Exception as e:
        print(f"\nMCTS Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_entei_vs_tyranitar()
    
    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)