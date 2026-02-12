"""
Turn-by-turn MCTS visualization that shows heuristic scores recalculated at each node.

This fixes the issue where cached/stale heuristic values were shown instead of
the actual heuristic considering current boosts and state.
"""

from typing import Any, Callable, Dict, List


def visualize_turn_by_turn(
    root,
    max_depth: int = 3,
    score_move_fn: Callable = None,
    score_switch_fn: Callable = None,
    battle: Any = None,
    ctx_me: Any = None,
    ctx_opp: Any = None,
):
    """
    Visualize MCTS tree with turn-by-turn breakdown and RECALCULATED heuristics.
    
    Args:
        root: Root MCTS node
        max_depth: Maximum depth to display
        score_move_fn: Function to score moves (for recalculating H)
        score_switch_fn: Function to score switches (for recalculating H)
        battle: Battle object (for recalculating H)
        ctx_me: Our context (for recalculating H)
        ctx_opp: Opponent context (for recalculating H)
    """
    
    print("\n" + "=" * 80)
    print("TURN-BY-TURN MCTS VISUALIZATION")
    print("=" * 80)
    
    # Count nodes per depth
    def count_depth(node, depth, counts):
        if depth not in counts:
            counts[depth] = 0
        counts[depth] += 1
        
        if node.children and depth < max_depth:
            for child in node.children.values():
                count_depth(child, depth + 1, counts)
    
    depth_counts = {}
    count_depth(root, 0, depth_counts)
    
    print("\nNodes per depth level:")
    for d in sorted(depth_counts.keys()):
        print(f"  Depth {d}: {depth_counts[d]} nodes")
    
    total = sum(depth_counts.values())
    print(f"\nTotal nodes in tree: {total}")
    
    print("\n" + "=" * 80)
    print("TURN-BY-TURN BREAKDOWN")
    print("=" * 80)
    
    print("\nLegend:")
    print("  N = Visit count (how many times MCTS explored this path)")
    print("  Q = Quality value (average reward from this node)")
    print("  Prior = Initial policy probability")
    print("  H = Heuristic score (raw evaluation RECALCULATED at this state)")
    print("  >> = Action taken")
    print("  [DMG] = Damage dealt")
    print("  [KO] = Pokemon fainted")
    print("  [AUTO-SWITCH] = Forced switch due to faint")
    print("  [CRIT] = Critical hit")
    print("  [MISS] = Move missed")
    
    # Print root
    state = root.state
    me = state.my_active
    opp = state.opp_active
    
    my_species = getattr(me, 'species', 'unknown')
    opp_species = getattr(opp, 'species', 'unknown')
    
    print(f"\n[D0] Root State │ {my_species} vs {opp_species}")
    print(f"     N={root.N} visits, Q={root.Q:+.3f}")
    
    # Show root actions with RECALCULATED heuristics
    if root.children:
        print("\nRoot action scores:")
        
        actions_with_scores = []
        for action, child in root.children.items():
            kind, obj = action[0], action[1]
            name = getattr(obj, 'id', getattr(obj, 'species', 'unknown'))
            
            # RECALCULATE heuristic at root state
            h_score = recalculate_heuristic(
                state, action, score_move_fn, score_switch_fn
            )
            
            actions_with_scores.append((action, child, name, h_score))
        
        # Sort by visit count
        actions_with_scores.sort(key=lambda x: x[1].N, reverse=True)
        
        for action, child, name, h_score in actions_with_scores:
            kind = action[0]
            pct = 100.0 * child.N / root.N if root.N > 0 else 0
            print(f"  [{kind.upper()}] {name:20s} | N={child.N:3d} ({pct:5.1f}%) "
                  f"Q={child.Q:+.3f} Prior={child.prior:.4f} H={h_score:+.3f}")
        
        # Show top 3 in detail
        print("\nExploring top 3 actions in detail:")
        for i, (action, child, name, h_score) in enumerate(actions_with_scores[:3]):
            print_action_subtree(
                action, child, root.state, 0, max_depth,
                score_move_fn, score_switch_fn, is_last=(i == len(actions_with_scores[:3]) - 1)
            )
        
        if len(actions_with_scores) > 3:
            print(f"... and {len(actions_with_scores) - 3} more actions not shown in detail")


def recalculate_heuristic(state, action, score_move_fn, score_switch_fn):
    """Recalculate heuristic score at the given state with current boosts/status."""
    if not score_move_fn or not score_switch_fn:
        return 0.0
    
    try:
        kind, obj = action[0], action[1]
        
        # Use patched boosts and status when recalculating
        with state._patched_status(), state._patched_boosts():
            if kind == "move":
                return float(score_move_fn(obj, state.battle, state.ctx_me))
            else:
                return float(score_switch_fn(obj, state.battle, state.ctx_me))
    except Exception as e:
        return 0.0


def print_action_subtree(action, node, parent_state, depth, max_depth, score_move_fn, score_switch_fn, is_last=False):
    """Print an action and its resulting subtree."""
    
    kind, obj = action[0], action[1]
    name = getattr(obj, 'id', getattr(obj, 'species', 'unknown'))
    
    # Determine what happened
    state = node.state
    events = get_turn_events(parent_state, state, action)
    
    # Print action
    prefix = "└─" if is_last else "├─"
    print(f"\n{prefix} We use: {name}")
    
    # Print events
    for event in events:
        print(f"   {event}")
    
    # Print resulting state with boosts
    me = state.my_active
    opp = state.opp_active
    my_hp = state.my_active_hp()
    opp_hp = state.opp_active_hp()
    
    my_species = getattr(me, 'species', 'unknown')
    opp_species = getattr(opp, 'species', 'unknown')
    
    # Show boosts
    my_boosts = state.my_boosts.get(id(me), {})
    my_boost_str = ""
    if my_boosts and any(v != 0 for v in my_boosts.values()):
        parts = [f"{s[:2]}{v:+d}" for s, v in my_boosts.items() if v != 0]
        my_boost_str = f"[{','.join(parts)}]"
    
    opp_boosts = state.opp_boosts.get(id(opp), {})
    opp_boost_str = ""
    if opp_boosts and any(v != 0 for v in opp_boosts.values()):
        parts = [f"{s[:2]}{v:+d}" for s, v in opp_boosts.items() if v != 0]
        opp_boost_str = f"[{','.join(parts)}]"
    
    child_prefix = "│  " if not is_last else "   "
    print(f"{child_prefix} [{my_species} {my_hp:.0%}{my_boost_str} vs {opp_species} {opp_hp:.0%}{opp_boost_str}] "
          f"N={node.N} Q={node.Q:+.3f}")
    
    # Show child actions if we have room
    if node.children and depth + 1 < max_depth:
        print(f"{child_prefix}")
        print(f"{child_prefix}Available actions at this node:")
        
        child_actions = []
        for child_action, child_node in node.children.items():
            child_kind, child_obj = child_action[0], child_action[1]
            child_name = getattr(child_obj, 'id', getattr(child_obj, 'species', 'unknown'))
            
            # RECALCULATE heuristic at this state (with current boosts!)
            h_score = recalculate_heuristic(
                state, child_action, score_move_fn, score_switch_fn
            )
            
            child_actions.append((child_action, child_node, child_name, h_score))
        
        # Sort by visits
        child_actions.sort(key=lambda x: x[1].N, reverse=True)
        
        for child_action, child_node, child_name, h_score in child_actions:
            child_kind = child_action[0]
            pct = 100.0 * child_node.N / node.N if node.N > 0 else 0
            print(f"{child_prefix}  [{child_kind.upper()}] {child_name:20s} | "
                  f"N={child_node.N:3d} ({pct:5.1f}%) Q={child_node.Q:+.3f} "
                  f"Prior={child_node.prior:.4f} H={h_score:+.3f}")
        
        # Recurse into top 3
        print(f"{child_prefix}")
        print(f"{child_prefix}Exploring top 3 actions:")
        print(f"{child_prefix}")
        
        for i, (child_action, child_node, child_name, h_score) in enumerate(child_actions[:3]):
            print_action_subtree(
                child_action, child_node, state, depth + 1, max_depth,
                score_move_fn, score_switch_fn, is_last=(i == 2)
            )
        
        if len(child_actions) > 3:
            print(f"{child_prefix}... and {len(child_actions) - 3} more actions not shown in detail")


def get_turn_events(parent_state, child_state, action):
    """Get a list of events that occurred during this turn."""
    events = []
    
    kind, obj = action[0], action[1]
    move_name = getattr(obj, 'id', getattr(obj, 'species', 'unknown'))
    
    parent_my_hp = parent_state.my_active_hp()
    child_my_hp = child_state.my_active_hp()
    parent_opp_hp = parent_state.opp_active_hp()
    child_opp_hp = child_state.opp_active_hp()
    
    parent_me = parent_state.my_active
    child_me = child_state.my_active
    parent_opp = parent_state.opp_active
    child_opp = child_state.opp_active
    
    # DEBUG: Always show HP changes
    parent_me_name = getattr(parent_me, 'species', 'unknown')
    child_me_name = getattr(child_me, 'species', 'unknown')
    parent_opp_name = getattr(parent_opp, 'species', 'unknown')
    child_opp_name = getattr(child_opp, 'species', 'unknown')
    
    # Track what happened
    we_switched_pokemon = (parent_me != child_me)
    opp_switched_pokemon = (parent_opp != child_opp)
    
    # Our action happened first (or we're looking at the result)
    if kind == "move":
        events.append(f"[ACTION] We used {move_name.upper()}")
    elif kind == "switch":
        events.append(f"[ACTION] We switched to {move_name}")
        we_switched_pokemon = True  # Mark this as intentional
    
    # Show before/after HP for debugging
    events.append(f"[DEBUG] Before: {parent_me_name} {parent_my_hp:.0%} vs {parent_opp_name} {parent_opp_hp:.0%}")
    events.append(f"[DEBUG] After:  {child_me_name} {child_my_hp:.0%} vs {child_opp_name} {child_opp_hp:.0%}")
    
    # Damage dealt to opponent (if they stayed in)
    if not opp_switched_pokemon:
        opp_damage = parent_opp_hp - child_opp_hp
        if opp_damage > 0.01:
            events.append(f"[DMG] Dealt {opp_damage:.1%} to {parent_opp_name}")
            
            # Did we KO them?
            if child_opp_hp <= 0.01:
                events.append(f"[KO] {parent_opp_name} fainted!")
    
    # Opponent's response
    if opp_switched_pokemon:
        # They switched out
        if parent_opp_hp <= 0.01:
            # Forced switch (they were KO'd)
            events.append(f"[FORCE-SWITCH] Opponent sent out {child_opp_name} (was KO'd)")
        else:
            # Voluntary switch
            events.append(f"[SWITCH] Opponent switched to {child_opp_name}")
    
    # Damage taken from opponent (only if we stayed in)
    if not we_switched_pokemon or kind == "switch":
        my_damage = parent_my_hp - child_my_hp
        if my_damage > 0.01:
            events.append(f"[DMG] Took {my_damage:.1%} damage")
            
            # Did we die?
            if child_my_hp <= 0.01:
                events.append(f"[KO] {parent_me_name} fainted!")
    
    # Our forced switch (if we died)
    if we_switched_pokemon and kind != "switch":
        if parent_my_hp <= 0.01:
            events.append(f"[FORCE-SWITCH] Sent out {child_me_name} (was KO'd)")
        else:
            events.append(f"[FORCE-SWITCH] Sent out {child_me_name} (HP was {parent_my_hp:.0%})")
    
    # Check for crit/miss in events
    if hasattr(child_state, 'events') and child_state.events:
        for event in child_state.events:
            event_str = str(event)
            if 'CRIT' in event_str and move_name in event_str:
                events.append("[CRIT]")
                break
        
        for event in child_state.events:
            event_str = str(event)
            if 'MISS' in event_str and move_name in event_str:
                events.append("[MISS]")
                break
    
    return events if events else ["[No visible effects]"]