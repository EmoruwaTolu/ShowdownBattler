"""
MCTS Tree Visualizer for Pokemon Battle Bot
============================================

A comprehensive visualization toolkit for MCTS showing step-by-step action history,
HP changes, and node statistics.

This file contains:
1. Core visualizer functions
2. Test/demo code with mock data
3. Integration examples

Usage:
------
# After MCTS search:
picked, stats = search(..., return_stats=True, return_tree=True)
root = stats["root"]
print(visualize_search_result(root, picked, stats))

# Or quick tree view:
print(format_tree_detailed(root, max_depth=2, top_k=3))

Run Tests:
----------
python mcts_visualizer.py
"""

from __future__ import annotations
from typing import Any, List, Dict, Optional, Tuple
from dataclasses import dataclass, field

Action = Tuple[str, Any]


# ============================================================================
# CORE VISUALIZER CODE
# ============================================================================

@dataclass
class StepInfo:
    """Information about a single step in the tree traversal"""
    depth: int
    node_id: str
    
    # Action taken
    action_type: str  # "move" or "switch"
    action_name: str
    outcome: Optional[str] = None  # For hybrid expansion: "hit", "crit", "miss"
    
    # Node statistics
    visits: int = 0
    q_value: float = 0.0
    prior: float = 0.0
    
    # State information
    my_pokemon: str = ""
    my_hp: float = 0.0
    my_status: str = "none"
    opp_pokemon: str = ""
    opp_hp: float = 0.0
    opp_status: str = "none"
    
    # What happened this turn
    events: List[str] = None
    
    def __post_init__(self):
        if self.events is None:
            self.events = []


def extract_state_info(node: Any) -> Dict[str, Any]:
    """Extract battle state information from a node"""
    state = node.state
    
    info = {
        "my_pokemon": "Unknown",
        "my_hp": 0.0,
        "my_status": "none",
        "opp_pokemon": "Unknown",
        "opp_hp": 0.0,
        "opp_status": "none",
        "events": []
    }
    
    try:
        # Extract Pokemon names
        info["my_pokemon"] = getattr(state.my_active, "species", 
                                     getattr(state.my_active, "name", "Unknown"))
        info["opp_pokemon"] = getattr(state.opp_active, "species",
                                      getattr(state.opp_active, "name", "Unknown"))
        
        # Extract HP
        my_id = id(state.my_active)
        opp_id = id(state.opp_active)
        info["my_hp"] = float(state.my_hp.get(my_id, 0.0))
        info["opp_hp"] = float(state.opp_hp.get(opp_id, 0.0))
        
        # Extract status
        my_st = state.my_status.get(my_id)
        opp_st = state.opp_status.get(opp_id)
        
        def format_status(status):
            if status is None:
                return "none"
            return str(status).split(".")[-1].lower()
        
        info["my_status"] = format_status(my_st)
        info["opp_status"] = format_status(opp_st)
        
        # Extract events if available
        if hasattr(state, "events") and state.events:
            info["events"] = list(state.events)
            
    except Exception as e:
        pass
    
    return info


def action_to_string(action: Action) -> Tuple[str, str, Optional[str]]:
    """
    Convert an action to (type, name, outcome) tuple
    
    Returns:
        (action_type, action_name, outcome)
    """
    if len(action) >= 3:
        kind, obj, outcome = action[0], action[1], action[2]
    else:
        kind, obj = action[0], action[1]
        outcome = None
    
    if kind == "move":
        name = getattr(obj, "id", getattr(obj, "name", "unknown_move"))
        return ("move", name, outcome)
    elif kind == "switch":
        name = getattr(obj, "species", getattr(obj, "name", "unknown_pokemon"))
        return ("switch", name, outcome)
    else:
        return (kind, str(obj), outcome)


def collect_path_history(
    root: Any,
    path: List[Any],
    actions: List[Action]
) -> List[StepInfo]:
    """
    Collect step-by-step history from a path through the tree.
    
    Args:
        root: Root node
        path: List of nodes from root to leaf
        actions: List of actions taken (one less than path length)
    
    Returns:
        List of StepInfo objects describing each step
    """
    history = []
    
    # Add root state
    root_state = extract_state_info(root)
    root_step = StepInfo(
        depth=0,
        node_id="D0",
        action_type="root",
        action_name="Initial State",
        visits=root.N,
        q_value=root.Q,
        prior=1.0,
        my_pokemon=root_state["my_pokemon"],
        my_hp=root_state["my_hp"],
        my_status=root_state["my_status"],
        opp_pokemon=root_state["opp_pokemon"],
        opp_hp=root_state["opp_hp"],
        opp_status=root_state["opp_status"],
        events=root_state["events"]
    )
    history.append(root_step)
    
    # Add each step in the path
    for i, (action, node) in enumerate(zip(actions, path[1:]), 1):
        action_type, action_name, outcome = action_to_string(action)
        state_info = extract_state_info(node)
        
        step = StepInfo(
            depth=i,
            node_id=f"D{i}",
            action_type=action_type,
            action_name=action_name,
            outcome=outcome,
            visits=node.N,
            q_value=node.Q,
            prior=node.prior,
            my_pokemon=state_info["my_pokemon"],
            my_hp=state_info["my_hp"],
            my_status=state_info["my_status"],
            opp_pokemon=state_info["opp_pokemon"],
            opp_hp=state_info["opp_hp"],
            opp_status=state_info["opp_status"],
            events=state_info["events"]
        )
        history.append(step)
    
    return history


def format_step_table(history: List[StepInfo], show_events: bool = True) -> str:
    """Format the step history as a nice ASCII table"""
    lines = []
    
    # Header
    lines.append("=" * 120)
    lines.append("MCTS TREE SEARCH PATH")
    lines.append("=" * 120)
    lines.append("")
    
    # Column headers
    header = (
        f"{'Step':<6} {'Action':<20} {'Out':<8} "
        f"{'N':<6} {'Q':<8} {'P':<7} "
        f"{'My Mon':<15} {'HP':<6} {'Status':<8} "
        f"{'Opp Mon':<15} {'HP':<6} {'Status':<8}"
    )
    lines.append(header)
    lines.append("-" * 120)
    
    # Each step
    for step in history:
        action_str = f"{step.action_type}: {step.action_name}"
        outcome_str = step.outcome or ""
        
        row = (
            f"{step.node_id:<6} "
            f"{action_str:<20} "
            f"{outcome_str:<8} "
            f"{step.visits:<6} "
            f"{step.q_value:+.3f}   "
            f"{step.prior:.3f}  "
            f"{step.my_pokemon:<15} "
            f"{step.my_hp:.2f}  "
            f"{step.my_status:<8} "
            f"{step.opp_pokemon:<15} "
            f"{step.opp_hp:.2f}  "
            f"{step.opp_status:<8}"
        )
        lines.append(row)
        
        # Show events if requested and available
        if show_events and step.events:
            for event in step.events[-2:]:  # Show last 2 events max
                lines.append(f"       └─ Event: {event}")
    
    lines.append("=" * 120)
    
    return "\n".join(lines)


def format_step_tree_style(history: List[StepInfo]) -> str:
    """Format the step history in a tree-like visual style"""
    lines = []
    lines.append("")
    lines.append("Tree Path (step-by-step):")
    lines.append("")
    
    for i, step in enumerate(history):
        # Indent based on depth
        indent = "  " * step.depth
        prefix = "└─" if i == len(history) - 1 else "├─"
        
        if step.action_type == "root":
            # Root node
            lines.append(
                f"[{step.node_id}] N={step.visits:4d} Q={step.q_value:+.3f} "
                f"{step.my_pokemon} {step.my_hp:.2f} vs {step.opp_pokemon} {step.opp_hp:.2f}"
            )
        else:
            # Action node
            action_str = f"{step.action_type} {step.action_name}"
            if step.outcome:
                action_str += f" [{step.outcome}]"
            
            lines.append(
                f"{indent}{prefix} {action_str:<30} "
                f"[{step.node_id}] N={step.visits:4d} Q={step.q_value:+.3f} "
                f"{step.my_pokemon} {step.my_hp:.2f} vs {step.opp_pokemon} {step.opp_hp:.2f}"
            )
            
            # Add events
            if step.events:
                event_indent = "  " * (step.depth + 1)
                for event in step.events[-2:]:
                    lines.append(f"{event_indent}• {event}")
    
    return "\n".join(lines)


def visualize_simulation(
    root: Any,
    path: List[Any],
    actions: List[Action],
    *,
    style: str = "table",
    show_events: bool = True
) -> str:
    """
    Main visualization function for a single MCTS simulation/rollout.
    
    Args:
        root: Root node of the tree
        path: List of nodes visited (including root)
        actions: List of actions taken (one less than path)
        style: "table" or "tree" format
        show_events: Whether to show event descriptions
    """
    history = collect_path_history(root, path, actions)
    
    if style == "tree":
        return format_step_tree_style(history)
    else:
        return format_step_table(history, show_events=show_events)


def format_children_summary(
    node: Any,
    max_children: int = 10,
    sort_by: str = "visits"
) -> str:
    """
    Format a summary of a node's children.
    
    Args:
        node: Node to summarize children for
        max_children: Maximum number of children to show
        sort_by: "visits" or "q" or "combined"
    """
    if not hasattr(node, 'children') or not node.children:
        return "No children to display"
    
    lines = []
    lines.append("")
    lines.append(f"Node Children (showing top {max_children}):")
    lines.append("")
    
    # Sort children
    if sort_by == "visits":
        sorted_children = sorted(
            node.children.items(),
            key=lambda x: x[1].N,
            reverse=True
        )
    elif sort_by == "q":
        sorted_children = sorted(
            node.children.items(),
            key=lambda x: x[1].Q,
            reverse=True
        )
    else:  # combined
        sorted_children = sorted(
            node.children.items(),
            key=lambda x: (x[1].N, x[1].Q),
            reverse=True
        )
    
    # Show top N children
    for i, (action, child) in enumerate(sorted_children[:max_children], 1):
        action_type, action_name, outcome = action_to_string(action)
        state_info = extract_state_info(child)
        
        action_str = f"{action_type} {action_name}"
        if outcome:
            action_str += f" [{outcome}]"
        
        line = (
            f"  {i:2d}. {action_str:<25} "
            f"N={child.N:4d} Q={child.Q:+.3f} P={child.prior:.3f}  "
            f"{state_info['my_pokemon']} {state_info['my_hp']:.2f} vs "
            f"{state_info['opp_pokemon']} {state_info['opp_hp']:.2f}"
        )
        lines.append(line)
    
    return "\n".join(lines)


def format_tree_detailed(root, max_depth: int = 3, top_k: int = 3) -> str:
    """
    Format tree in a style similar to the screenshot.
    Shows: [D#] N=### Q=+#.### Pokemon1 HP vs Pokemon2 HP
    """
    lines = []
    
    def recurse(node, depth: int, prefix: str, is_last: bool):
        # Format this node
        state = extract_state_info(node)
        node_str = (
            f"[D{depth}] N={node.N:4d} Q={node.Q:+.3f}  "
            f"{state['my_pokemon'][:12]:<12} {state['my_hp']:.2f} vs "
            f"{state['opp_pokemon'][:15]:<15} {state['opp_hp']:.2f}"
        )
        
        if depth == 0:
            lines.append(f"└─ {node_str}")
        
        if depth >= max_depth or not node.children:
            return
        
        # Get top children by visits
        children = sorted(
            node.children.items(),
            key=lambda x: (x[1].N, x[1].Q),
            reverse=True
        )[:top_k]
        
        child_prefix = prefix + ("   " if depth == 0 else ("   " if is_last else "│  "))
        
        for i, (action, child) in enumerate(children):
            is_last_child = (i == len(children) - 1)
            action_type, action_name, outcome = action_to_string(action)
            
            # Action line
            action_str = f"{action_type} {action_name}"
            if outcome:
                action_str += f" [{outcome}]"
            
            branch = "└─" if is_last_child else "├─"
            lines.append(f"{child_prefix}{branch} {action_str}")
            
            # Child node line
            next_prefix = child_prefix + ("   " if is_last_child else "│  ")
            child_state = extract_state_info(child)
            
            child_node_str = (
                f"[D{depth+1}] N={child.N:4d} Q={child.Q:+.3f}  "
                f"{child_state['my_pokemon'][:12]:<12} {child_state['my_hp']:.2f} vs "
                f"{child_state['opp_pokemon'][:15]:<15} {child_state['opp_hp']:.2f}"
            )
            
            lines.append(f"{next_prefix}└─ {child_node_str}")
            
            # Recurse into this child
            recurse(child, depth + 1, next_prefix, is_last_child)
        
        # Show "and X more" if applicable
        if len(node.children) > top_k:
            remaining = len(node.children) - top_k
            lines.append(f"{child_prefix}   ... and {remaining} more children")
    
    recurse(root, 0, "", True)
    return "\n".join(lines)


def visualize_search_result(
    root: Any,
    best_action: Action,
    *,
    show_path: bool = True,
    show_children: bool = True,
    path_style: str = "table",
    max_children: int = 10
) -> str:
    """
    Complete visualization of an MCTS search result.
    
    Args:
        root: Root node
        best_action: The action selected by MCTS
        show_path: Show the best path through the tree
        show_children: Show root's children summary
        path_style: "table" or "tree" for path visualization
        max_children: Number of children to show
    """
    lines = []
    
    # Header
    lines.append("=" * 120)
    lines.append("MCTS SEARCH RESULT")
    lines.append("=" * 120)
    lines.append("")
    
    # Best action
    action_type, action_name, outcome = action_to_string(best_action)
    lines.append(f"Selected Action: {action_type} {action_name}")
    if outcome:
        lines.append(f"Outcome: {outcome}")
    lines.append("")
    
    # Root children summary
    if show_children:
        lines.append(format_children_summary(root, max_children=max_children))
        lines.append("")
    
    # Best path (if child exists)
    if show_path and best_action in root.children:
        best_child = root.children[best_action]
        # Simple path: root -> best child
        path = [root, best_child]
        actions = [best_action]
        
        lines.append(visualize_simulation(
            root, path, actions,
            style=path_style,
            show_events=True
        ))
    
    return "\n".join(lines)


def compare_top_actions(root, n: int = 5):
    """Compare the top N actions at the root with detailed stats"""
    print("\n" + "=" * 120)
    print(f" TOP {n} ACTIONS COMPARISON")
    print("=" * 120)
    print()
    
    # Sort by visits
    children = sorted(
        root.children.items(),
        key=lambda x: (x[1].N, x[1].Q),
        reverse=True
    )[:n]
    
    for i, (action, child) in enumerate(children, 1):
        action_type, action_name, outcome = action_to_string(action)
        state = extract_state_info(child)
        
        print(f"{i}. {action_type.upper()}: {action_name}" + (f" [{outcome}]" if outcome else ""))
        print(f"   Statistics:")
        print(f"      Visits: {child.N}")
        print(f"      Q-value: {child.Q:+.4f}")
        print(f"      Prior: {child.prior:.4f}")
        print(f"      Win rate: {(child.Q + 1) / 2 * 100:.1f}%")
        print(f"   Resulting state:")
        print(f"      My Pokemon: {state['my_pokemon']} (HP: {state['my_hp']:.2f}, Status: {state['my_status']})")
        print(f"      Opp Pokemon: {state['opp_pokemon']} (HP: {state['opp_hp']:.2f}, Status: {state['opp_status']})")
        
        if state['events']:
            print(f"   Events:")
            for event in state['events'][-3:]:
                print(f"      • {event}")
        
        print()


# ============================================================================
# TEST/DEMO CODE
# ============================================================================

@dataclass
class MockPokemon:
    """Mock Pokemon for testing"""
    species: str
    name: str
    hp: float = 1.0
    
    def __hash__(self):
        return hash(self.species)


@dataclass  
class MockBattleState:
    """Mock battle state for testing"""
    my_active: MockPokemon
    opp_active: MockPokemon
    my_hp: Dict[int, float] = field(default_factory=dict)
    opp_hp: Dict[int, float] = field(default_factory=dict)
    my_status: Dict[int, Any] = field(default_factory=dict)
    opp_status: Dict[int, Any] = field(default_factory=dict)
    my_team: List[MockPokemon] = field(default_factory=list)
    opp_team: List[MockPokemon] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    
    def is_terminal(self):
        return False


@dataclass
class MockNode:
    """Mock MCTS node for testing"""
    state: MockBattleState
    N: int = 0
    W: float = 0.0
    prior: float = 1.0
    children: Dict[Tuple, 'MockNode'] = field(default_factory=dict)
    
    @property
    def Q(self):
        return self.W / self.N if self.N > 0 else 0.0


@dataclass(frozen=True)
class MockMove:
    """Mock move for testing"""
    id: str
    name: str


def create_test_tree():
    """Create a mock tree for testing (similar to screenshot)"""
    
    # Pokemon
    garchomp = MockPokemon("Garchomp", "Garchomp")
    weavile = MockPokemon("Weavile", "Weavile")
    toxapex = MockPokemon("Toxapex", "Toxapex")
    landorus = MockPokemon("Landorus-Therian", "Landorus")
    clefable = MockPokemon("Clefable", "Clefable")
    
    # Root state
    root_state = MockBattleState(
        my_active=garchomp,
        opp_active=landorus,
        my_team=[garchomp, weavile, toxapex],
        opp_team=[landorus, clefable]
    )
    root_state.my_hp[id(garchomp)] = 0.80
    root_state.opp_hp[id(landorus)] = 0.65
    
    root = MockNode(state=root_state, N=400, W=192.0, prior=1.0)
    
    # Child 1: Switch to Weavile
    switch_state = MockBattleState(
        my_active=weavile,
        opp_active=clefable,
        my_team=[garchomp, weavile, toxapex],
        opp_team=[landorus, clefable],
        events=["Switched to Weavile", "Clefable used Moonblast"]
    )
    switch_state.my_hp[id(weavile)] = 0.85
    switch_state.opp_hp[id(clefable)] = 0.40
    
    switch_node = MockNode(state=switch_state, N=343, W=161.0, prior=0.25)
    
    # Moves
    swords_dance = MockMove("swordsdance", "Swords Dance")
    ice_shard = MockMove("iceshard", "Ice Shard")
    triple_axel = MockMove("tripleaxel", "Triple Axel")
    
    # Child 1.1: Swords Dance
    sd_state = MockBattleState(
        my_active=garchomp,
        opp_active=clefable,
        my_team=[garchomp, weavile, toxapex],
        opp_team=[landorus, clefable],
        events=["Garchomp used Swords Dance", "Garchomp's Attack rose sharply!"]
    )
    sd_state.my_hp[id(garchomp)] = 0.80
    sd_state.opp_hp[id(clefable)] = 0.40
    
    sd_node = MockNode(state=sd_state, N=300, W=141.3, prior=0.32)
    switch_node.children[("move", swords_dance)] = sd_node
    
    # Child 1.2: Ice Shard  
    is_state = MockBattleState(
        my_active=garchomp,
        opp_active=clefable,
        my_team=[garchomp, weavile, toxapex],
        opp_team=[landorus, clefable],
        events=["Garchomp used Ice Shard", "It's not very effective..."]
    )
    is_state.my_hp[id(garchomp)] = 0.80
    is_state.opp_hp[id(clefable)] = 0.22
    
    is_node = MockNode(state=is_state, N=13, W=5.1, prior=0.15)
    switch_node.children[("move", ice_shard)] = is_node
    
    # Child 1.3: Triple Axel
    ta_state = MockBattleState(
        my_active=garchomp,
        opp_active=clefable,
        my_team=[garchomp, weavile, toxapex],
        opp_team=[landorus, clefable],
        events=["Garchomp used Triple Axel", "Hit 3 times!"]
    )
    ta_state.my_hp[id(garchomp)] = 0.80
    ta_state.opp_hp[id(clefable)] = 0.31
    
    ta_node = MockNode(state=ta_state, N=11, W=5.1, prior=0.12)
    switch_node.children[("move", triple_axel)] = ta_node
    
    root.children[("switch", weavile)] = switch_node
    
    # Child 2: Outrage
    outrage = MockMove("outrage", "Outrage")
    outrage_state = MockBattleState(
        my_active=garchomp,
        opp_active=clefable,
        my_team=[garchomp, weavile, toxapex],
        opp_team=[landorus, clefable],
        events=["Garchomp used Outrage", "It's super effective!"]
    )
    outrage_state.my_hp[id(garchomp)] = 0.80
    outrage_state.opp_hp[id(clefable)] = 0.15
    
    outrage_node = MockNode(state=outrage_state, N=29, W=16.2, prior=0.18)
    root.children[("move", outrage)] = outrage_node
    
    # Child 3: Switch to Toxapex
    tox_state = MockBattleState(
        my_active=toxapex,
        opp_active=clefable,
        my_team=[garchomp, weavile, toxapex],
        opp_team=[landorus, clefable],
        events=["Switched to Toxapex"]
    )
    tox_state.my_hp[id(toxapex)] = 0.90
    tox_state.opp_hp[id(clefable)] = 0.40
    
    tox_node = MockNode(state=tox_state, N=15, W=8.4, prior=0.12)
    root.children[("switch", toxapex)] = tox_node
    
    return root


def run_tests():
    """Run all visualization tests"""
    print("\n" + "=" * 120)
    print("MCTS VISUALIZER - TEST SUITE")
    print("=" * 120)
    
    print("\nCreating mock MCTS tree...")
    root = create_test_tree()
    
    # Get best action
    best_action = max(root.children.items(), key=lambda x: x[1].N)[0]
    
    print("\n" + "=" * 120)
    print("TEST 1: Children Summary")
    print("=" * 120)
    print(format_children_summary(root, max_children=3, sort_by="visits"))
    
    print("\n" + "=" * 120)
    print("TEST 2: Tree Format (like screenshot)")
    print("=" * 120)
    print(format_tree_detailed(root, max_depth=2, top_k=3))
    
    print("\n" + "=" * 120)
    print("TEST 3: Compare Top Actions")
    print("=" * 120)
    compare_top_actions(root, n=3)
    
    print("\n" + "=" * 120)
    print("TEST 4: Simulation Path (Table)")
    print("=" * 120)
    
    best_child = root.children[best_action]
    path = [root, best_child]
    actions = [best_action]
    
    print(visualize_simulation(root, path, actions, style="table"))
    
    print("\n" + "=" * 120)
    print("TEST 5: Simulation Path (Tree)")
    print("=" * 120)
    print(visualize_simulation(root, path, actions, style="tree"))
    
    print("\n" + "=" * 120)
    print("TEST 6: Complete Search Result")
    print("=" * 120)
    print(visualize_search_result(
        root=root,
        best_action=best_action,
        show_path=True,
        show_children=True,
        path_style="table"
    ))
    
    print("\n" + "=" * 120)
    print("ALL TESTS PASSED! ✓")
    print("=" * 120)
    print("\nIntegration with your MCTS:")
    print("1. In your search call, add: return_stats=True, return_tree=True")
    print("2. Extract root: root = stats['root']")
    print("3. Visualize: print(format_tree_detailed(root, max_depth=2, top_k=3))")
    print()


if __name__ == "__main__":
    run_tests()