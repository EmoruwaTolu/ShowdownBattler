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


def visualize_tree_depth(root, max_depth=4, show_opponent_moves=False):
    """
    Visualize the MCTS tree showing depth and states.
    
    Args:
        root: Root node of MCTS tree
        max_depth: Maximum depth to visualize
        show_opponent_moves: If True, show detailed opponent move analysis
    """
    print("\n" + "=" * 80)
    print("MCTS TREE DEPTH VISUALIZATION")
    print("=" * 80)
    
    def get_state_summary(state):
        """Get a summary of the game state - works with your ShadowState"""
        me = state.my_active
        opp = state.opp_active
        
        my_hp = state.my_active_hp()
        opp_hp = state.opp_active_hp()
        
        # Get status
        my_status = state.my_active_status()
        opp_status = state.opp_active_status()
        
        # Get boosts
        my_boosts = state.my_boosts.get(id(me), {})
        opp_boosts = state.opp_boosts.get(id(opp), {})
        
        my_species = getattr(me, 'species', 'Unknown')
        opp_species = getattr(opp, 'species', 'Unknown')
        
        # Format my side
        my_str = f"{my_species} {my_hp:.2f}"
        
        # Add status
        if my_status:
            status_name = str(my_status).split('.')[-1].lower()[:3]
            my_str += f"({status_name})"
        
        # Add significant boosts (non-zero)
        my_boost_str = ""
        for stat in ['atk', 'def', 'spa', 'spd', 'spe']:
            val = my_boosts.get(stat, 0)
            if val != 0:
                my_boost_str += f"{stat[:2]}{val:+d} "
        
        if my_boost_str:
            my_str += f"[{my_boost_str.strip()}]"

        # Add volatile indicators
        my_vol = getattr(state, 'my_volatiles', {}).get(id(me), {})
        if my_vol.get('sleep_turns'):
            my_str += f"(slp:{my_vol['sleep_turns']})"
        if my_vol.get('confusion_turns'):
            my_str += "(cnf)"
        if my_vol.get('protect'):
            my_str += "(prt)"

        # Format opponent side
        opp_str = f"{opp_species} {opp_hp:.2f}"

        # Add status
        if opp_status:
            status_name = str(opp_status).split('.')[-1].lower()[:3]
            opp_str += f"({status_name})"

        # Add significant boosts
        opp_boost_str = ""
        for stat in ['atk', 'def', 'spa', 'spd', 'spe']:
            val = opp_boosts.get(stat, 0)
            if val != 0:
                opp_boost_str += f"{stat[:2]}{val:+d} "

        if opp_boost_str:
            opp_str += f"[{opp_boost_str.strip()}]"

        # Add volatile indicators
        opp_vol = getattr(state, 'opp_volatiles', {}).get(id(opp), {})
        if opp_vol.get('sleep_turns'):
            opp_str += f"(slp:{opp_vol['sleep_turns']})"
        if opp_vol.get('confusion_turns'):
            opp_str += "(cnf)"
        if opp_vol.get('protect'):
            opp_str += "(prt)"

        return f"{my_str} vs {opp_str}"
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
                
                # NEW: Add outcome annotations (hit/miss/crit)
                if outcome:
                    if outcome == 'crit':
                        action_str += " 💥"  # Crit indicator
                    elif outcome == 'hit':
                        action_str += " ✓"   # Hit indicator
                    elif outcome == 'miss':
                        action_str += " ✗"   # Miss indicator
                
                # Check for crit in debug events
                if hasattr(child.state, 'events') and child.state.events:
                    events = child.state.events
                    move_category = getattr(obj, 'category', None)
                    
                    # Only check CRIT for damaging moves
                    if move_category in [MoveCategory.PHYSICAL, MoveCategory.SPECIAL]:
                        for event in events:
                            event_str = str(event)
                            # Only show CRIT if it's our move
                            if f'CRIT me:{move_name}' in event_str:
                                action_str += " CRIT"
                                break
                    
                    # Check MISS for all moves (including STATUS moves that can miss)
                    for event in events:
                        event_str = str(event)
                        # Only show MISS if it's our move
                        if f'MISS me:{move_name}' in event_str:
                            action_str += " MISS"
                            break
                
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
    
    # If requested, show detailed opponent move analysis
    if show_opponent_moves:
        print_opponent_move_analysis(root)
    
    return depth_counts


def print_opponent_move_analysis(root):
    """
    Print detailed analysis of opponent moves for each of our moves.
    """
    print("\n" + "=" * 80)
    print("DETAILED OPPONENT MOVE ANALYSIS")
    print("=" * 80)
    
    # Get all our moves from root children
    our_moves = []
    for action, child in root.children.items():
        if action[0] == 'move':
            move_name = getattr(action[1], 'id', 'unknown')
            our_moves.append((move_name, action, child))
    
    # Sort by visits
    our_moves.sort(key=lambda x: x[2].N, reverse=True)
    
    for move_name, our_action, our_node in our_moves:
        print(f"\n{'='*80}")
        print(f"After we use: {move_name.upper()}")
        print(f"{'='*80}")
        print(f"Visits: {our_node.N:4d} ({our_node.N/root.N*100:5.1f}%)")
        print(f"Q-value: {our_node.Q:+.3f}")
        print(f"Prior: {our_node.prior:.4f}")
        
        # State after our move
        try:
            state = our_node.state
            print(f"\nState after {move_name}:")
            print(f"  Us:       {state.my_active.species} at {state.my_active_hp():.2f} HP", end="")
            my_status = state.my_active_status()
            if my_status:
                print(f" ({my_status})", end="")
            print()
            
            print(f"  Opponent: {state.opp_active.species} at {state.opp_active_hp():.2f} HP", end="")
            opp_status = state.opp_active_status()
            if opp_status:
                print(f" ({opp_status})", end="")
            print()
        except Exception as e:
            print(f"  [Error getting state: {e}]")
        
        # Check if terminal
        if not our_node.children:
            if our_node.state.is_terminal():
                print(f"\n  → Terminal state (game over)")
            else:
                print(f"\n  → Not expanded (only {our_node.N} visit{'s' if our_node.N != 1 else ''})")
            continue
        
        # Opponent's responses
        print(f"\n  Opponent's Responses ({len(our_node.children)} moves explored):")
        print(f"  {'-'*76}")
        
        opp_moves = sorted(
            our_node.children.items(),
            key=lambda x: x[1].N,
            reverse=True
        )
        
        for i, (opp_action, opp_node) in enumerate(opp_moves[:5], 1):
            opp_move_name = getattr(opp_action[1], 'id', 'unknown')
            opp_visits = opp_node.N
            opp_pct = (opp_visits / our_node.N * 100) if our_node.N > 0 else 0
            
            print(f"\n  {i}. {opp_move_name}")
            print(f"     Explored: {opp_visits:4d} times ({opp_pct:5.1f}%)")
            print(f"     Q-value: {opp_node.Q:+.3f}")
            
            # Result state
            try:
                result_state = opp_node.state
                print(f"     Result:")
                
                # Our status
                our_hp = result_state.my_active_hp()
                our_status = result_state.my_active_status()
                print(f"       Us:       {result_state.my_active.species} at {our_hp:.2f} HP", end="")
                if our_status:
                    print(f" ({our_status})", end="")
                if our_hp <= 0:
                    print(" ← FAINTED!", end="")
                print()
                
                # Opponent status
                opp_hp = result_state.opp_active_hp()
                opp_status_result = result_state.opp_active_status()
                print(f"       Opponent: {result_state.opp_active.species} at {opp_hp:.2f} HP", end="")
                if opp_status_result:
                    print(f" ({opp_status_result})", end="")
                if opp_hp <= 0:
                    print(" ← FAINTED!", end="")
                print()
                
                # Show if status changed
                if opp_status_result and not opp_status:
                    print(f"       → Opponent got {opp_status_result}!")
                if our_status and not my_status:
                    print(f"       → We got {our_status}!")
                
            except Exception as e:
                print(f"     [Error: {e}]")
            
            # Show our follow-ups if explored
            if opp_node.children and i <= 2:  # Only show for top 2 opponent moves
                followups = sorted(
                    opp_node.children.items(),
                    key=lambda x: x[1].N,
                    reverse=True
                )[:3]
                
                if followups:
                    print(f"     Our follow-ups:")
                    for followup_action, followup_node in followups:
                        followup_name = getattr(followup_action[1], 'id', 'unknown')
                        print(f"       • {followup_name}: {followup_node.N} visits, Q={followup_node.Q:+.3f}")


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
    
    # Visualize tree WITH opponent move analysis
    root = stats['root']
    depth_counts = visualize_tree_depth(root, max_depth=3, show_opponent_moves=True)
    
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