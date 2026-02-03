import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict
from unittest.mock import Mock

from poke_env.battle import MoveCategory, PokemonType
from bot.model.ctx import EvalContext
from bot.mcts.search import search, MCTSConfig

# Use the same mock setup
from test_real_heuristics import (
    create_mock_pokemon,
    create_mock_move,
    create_mock_battle,
    score_move,
    score_switch,
    estimate_damage_fraction,
)


def visualize_tree_depth(root, max_depth=4):
    """
    Visualize the MCTS tree showing depth and states.
    """
    print("\n" + "=" * 80)
    print("MCTS TREE DEPTH VISUALIZATION")
    print("=" * 80)
    
    def get_state_summary(state):
        """Get a summary of the game state - works with your ShadowState"""
        # Direct attributes (not methods)
        me = state.my_active
        opp = state.opp_active
        
        # Methods for HP
        my_hp = state.my_active_hp()
        opp_hp = state.opp_active_hp()
        
        my_species = getattr(me, 'species', 'Unknown')
        opp_species = getattr(opp, 'species', 'Unknown')
        
        return f"{my_species} {my_hp:.2f} vs {opp_species} {opp_hp:.2f}"
    
    def count_nodes_at_depth(node, current_depth, max_depth, counts):
        """Count nodes at each depth level"""
        if current_depth > max_depth:
            return
        
        if current_depth not in counts:
            counts[current_depth] = 0
        counts[current_depth] += 1
        
        for child in node.children.values():
            count_nodes_at_depth(child, current_depth + 1, max_depth, counts)
    
    def print_tree_recursive(node, depth, prefix, is_last, max_depth, show_states=True):
        """Recursively print tree structure"""
        if depth > max_depth:
            return
        
        # Current node info
        state_str = ""
        if show_states:
            try:
                state_str = get_state_summary(node.state)
            except Exception as e:
                state_str = f"[Error: {e}]"
        
        print(f"{prefix}{'└─' if is_last else '├─'} [D{depth}] N={node.N:3d} Q={node.Q:+.3f} {state_str}")
        
        # Print children
        if node.children and depth < max_depth:
            children = sorted(node.children.items(), key=lambda x: x[1].N, reverse=True)
            
            # Show top 3 children
            num_to_show = min(3, len(children))
            for i, (action, child) in enumerate(children[:num_to_show]):
                is_last_child = (i == len(children) - 1) or (i == num_to_show - 1)
                
                # Action description
                kind = action[0]
                obj = action[1]
                outcome = action[2] if len(action) > 2 else None
                
                move_name = getattr(obj, 'id', getattr(obj, 'species', 'unknown'))
                action_str = f"{kind} {move_name}"
                if outcome:
                    action_str += f" [{outcome}]"
                
                # Prepare prefix for child
                if is_last:
                    child_prefix = prefix + "    "
                else:
                    child_prefix = prefix + "│   "
                
                print(f"{child_prefix}{'└─' if is_last_child else '├─'} {action_str}")
                print_tree_recursive(
                    child, 
                    depth + 1, 
                    child_prefix + ("    " if is_last_child else "│   "), 
                    True, 
                    max_depth, 
                    show_states
                )
            
            if len(children) > num_to_show:
                print(f"{child_prefix}... and {len(children) - num_to_show} more children")
    
    # Count nodes at each depth
    depth_counts = {}
    count_nodes_at_depth(root, 0, max_depth, depth_counts)
    
    print("\nNodes per depth level:")
    for d in sorted(depth_counts.keys()):
        print(f"  Depth {d}: {depth_counts[d]} nodes")
    
    print(f"\nTotal nodes in tree: {sum(depth_counts.values())}")
    print()
    
    # Print tree structure
    print("Tree Structure (showing top 3 children per node):")
    print()
    print_tree_recursive(root, 0, "", True, max_depth, show_states=True)
    
    return depth_counts


def test_basic_tree():
    """
    Test to show tree depth with a simple scenario.
    """
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 25 + "BASIC TREE DEPTH" + " " * 36 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Create scenario
    garchomp = create_mock_pokemon(
        species="Garchomp",
        identifier="p1: Garchomp",
        hp_frac=0.85,
        types=(PokemonType.GROUND, PokemonType.DRAGON),
        stats={'hp': 183, 'atk': 182, 'def': 115, 'spa': 100, 'spd': 105, 'spe': 169},
    )
    
    rotom = create_mock_pokemon(
        species="Rotom",
        identifier="p2: Rotom",
        hp_frac=0.60,
        types=(PokemonType.ELECTRIC, PokemonType.WATER),
        stats={'hp': 127, 'atk': 85, 'def': 147, 'spa': 125, 'spd': 147, 'spe': 126},
    )
    
    # Moves
    earthquake = create_mock_move("earthquake", MoveCategory.PHYSICAL, PokemonType.GROUND, 100, 1.0)
    outrage = create_mock_move("outrage", MoveCategory.PHYSICAL, PokemonType.DRAGON, 120, 1.0)
    stoneedge = create_mock_move("stoneedge", MoveCategory.PHYSICAL, PokemonType.ROCK, 100, 0.8)
    
    garchomp.moves = {"earthquake": earthquake, "outrage": outrage, "stoneedge": stoneedge}
    
    hydropump = create_mock_move("hydropump", MoveCategory.SPECIAL, PokemonType.WATER, 110, 0.8)
    rotom.moves = {"hydropump": hydropump}
    
    team = {"p1: Garchomp": garchomp}
    opp_team = {"p2: Rotom": rotom}
    
    battle = create_mock_battle("p1: Garchomp", garchomp, "p2: Rotom", rotom, team, opp_team)
    battle.available_moves = [earthquake, outrage, stoneedge]
    
    ctx_me = EvalContext(me=garchomp, opp=rotom, battle=battle, cache={})
    ctx_opp = EvalContext(me=rotom, opp=garchomp, battle=battle, cache={})
    
    # Run MCTS with depth tracking
    print("\nRunning MCTS...")
    print("  - 200 simulations")
    print("  - max_depth=4")
    print("  - No hybrid expansion")
    
    cfg = MCTSConfig(
        num_simulations=200,
        max_depth=4,
        seed=42,
        use_hybrid_expansion=False,
    )
    
    picked, stats = search(
        battle=battle,
        ctx_me=ctx_me,
        ctx_opp=ctx_opp,
        score_move_fn=score_move,
        score_switch_fn=score_switch,
        dmg_fn=estimate_damage_fraction,
        cfg=cfg,
        opp_tau=0.0001,
        return_stats=True,
        return_tree=True,
    )
    
    # Visualize tree
    root = stats['root']
    depth_counts = visualize_tree_depth(root, max_depth=3)
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print(f"Root visits: {root.N}")
    print(f"Root Q-value: {root.Q:+.3f}")
    print()
    print("Interpretation:")
    print("  - Depth 0: Root (our turn)")
    print("  - Depth 1: After we choose a move")
    print("  - Depth 2: After opponent responds")
    print("  - Depth 3: After we respond again")
    print()
    print(f"Max depth reached: {max(depth_counts.keys())}")
    print(f"Total nodes explored: {sum(depth_counts.values())}")


def test_branching_tree():
    """
    Test tree depth WITH hybrid expansion.
    """
    print("\n\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 25 + "BRANCHING TREE" + " " * 38 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Stone Edge scenario
    garchomp = create_mock_pokemon(
        species="Garchomp",
        identifier="p1: Garchomp",
        hp_frac=0.85,
        types=(PokemonType.GROUND, PokemonType.DRAGON),
        stats={'hp': 183, 'atk': 182, 'def': 115, 'spa': 100, 'spd': 105, 'spe': 169},
    )
    
    moltres = create_mock_pokemon(
        species="Moltres",
        identifier="p2: Moltres",
        hp_frac=0.65,
        types=(PokemonType.FIRE, PokemonType.FLYING),
        stats={'hp': 165, 'atk': 120, 'def': 110, 'spa': 145, 'spd': 105, 'spe': 110},
    )
    
    stoneedge = create_mock_move("stoneedge", MoveCategory.PHYSICAL, PokemonType.ROCK, 100, 0.8)
    garchomp.moves = {"stoneedge": stoneedge}
    
    flamethrower = create_mock_move("flamethrower", MoveCategory.SPECIAL, PokemonType.FIRE, 90, 1.0)
    moltres.moves = {"flamethrower": flamethrower}
    
    team = {"p1: Garchomp": garchomp}
    opp_team = {"p2: Moltres": moltres}
    
    battle = create_mock_battle("p1: Garchomp", garchomp, "p2: Moltres", moltres, team, opp_team)
    battle.available_moves = [stoneedge]
    
    ctx_me = EvalContext(me=garchomp, opp=moltres, battle=battle, cache={})
    ctx_opp = EvalContext(me=moltres, opp=garchomp, battle=battle, cache={})
    
    print("\nRunning MCTS with HYBRID EXPANSION...")
    print("  - 150 simulations")
    print("  - max_depth=3")
    print("  - Stone Edge (80% acc) will branch")
    
    cfg = MCTSConfig(
        num_simulations=150,
        max_depth=3,
        seed=42,
        use_hybrid_expansion=True,
        branch_low_accuracy=True,
        low_accuracy_threshold=0.85,
        min_branch_probability=0.01,
    )
    
    picked, stats = search(
        battle=battle,
        ctx_me=ctx_me,
        ctx_opp=ctx_opp,
        score_move_fn=score_move,
        score_switch_fn=score_switch,
        dmg_fn=estimate_damage_fraction,
        cfg=cfg,
        opp_tau=0.0001,
        return_stats=True,
        return_tree=True,
    )
    
    # Visualize tree
    root = stats['root']
    depth_counts = visualize_tree_depth(root, max_depth=2)
    
    print("\n" + "=" * 80)
    print("BRANCHING ANALYSIS")
    print("=" * 80)
    print("Notice how Depth 1 has 3 children (one per outcome):")
    print("  - Stone Edge [hit+crit]: Low probability, high reward")
    print("  - Stone Edge [hit]: Most likely outcome")
    print("  - Stone Edge [miss]: Risk of missing")
    print()
    print("Each branch explores different game states!")


if __name__ == "__main__":
    test_basic_tree()
    test_branching_tree()