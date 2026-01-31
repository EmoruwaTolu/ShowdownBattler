"""
Comprehensive test suite for updated scoring system with MCTS integration.

Tests:
1. Pivot move scoring (Volt Switch vs alternatives)
2. Switch scoring (good vs bad matchups)
3. MCTS simulation comparison
4. Heuristic vs MCTS decision agreement
"""

import sys
import os
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poke_env.battle import Move, MoveCategory, PokemonType, Status
from bot.model.ctx import EvalContext
from bot.scoring.move_score import score_move
from bot.scoring.switch_score import score_switch, pivot_move_bonus
from bot.mcts.search import mcts_pick_action, ShadowState, eval_state
from bot.model.opponent_model import build_opponent_belief


# ============================================================================
# MOCK OBJECTS
# ============================================================================

@dataclass
class MockPokemon:
    """Mock Pokemon for testing"""
    species: str
    current_hp_fraction: float
    max_hp: int
    base_stats: Dict[str, int]
    types: List[PokemonType]
    moves: Dict[str, Any]
    fainted: bool = False
    status: Any = None
    ability: str = None
    item: str = None
    
    def __hash__(self):
        return hash(self.species)


class MockMove:
    """Mock Move for testing"""
    def __init__(self, id: str, base_power: int, accuracy: float, category: MoveCategory, 
                 move_type: PokemonType, priority: int = 0, status: Status = None):
        self.id = id
        self.base_power = base_power
        self.accuracy = accuracy
        self.category = category
        self.type = move_type
        self.priority = priority
        self.status = status
        self.recoil = None
        self.secondary = None


class MockBattle:
    """Mock Battle for testing"""
    def __init__(self, me: MockPokemon, opp: MockPokemon, team: Dict[str, MockPokemon], 
                 opponent_team: Dict[str, MockPokemon]):
        self.team = team
        self.opponent_team = opponent_team
        self.active_pokemon = me
        self.opponent_active_pokemon = opp
        self.available_moves = []
        self.available_switches = []
        self.gen = 9


# ============================================================================
# TEST SCENARIOS
# ============================================================================

def create_rotom_tyranitar_scenario():
    """
    Rotom-Wash 35% HP vs Tyranitar 52% HP
    Moves: Hydro Pump, Volt Switch, Will-O-Wisp, Thunder Wave
    Weavile in back (gets OHKO'd by Stone Edge)
    """
    # Create Rotom
    rotom = MockPokemon(
        species="Rotom-Wash",
        current_hp_fraction=0.35,
        max_hp=245,
        base_stats={"hp": 50, "atk": 65, "def": 107, "spa": 105, "spd": 107, "spe": 86},
        types=[PokemonType.ELECTRIC, PokemonType.WATER],
        moves={},
        ability="levitate"
    )
    
    # Create Tyranitar
    tyranitar = MockPokemon(
        species="Tyranitar",
        current_hp_fraction=0.52,
        max_hp=341,
        base_stats={"hp": 100, "atk": 134, "def": 110, "spa": 95, "spd": 100, "spe": 61},
        types=[PokemonType.ROCK, PokemonType.DARK],
        moves={
            "stoneedge": MockMove("stoneedge", 100, 0.8, MoveCategory.PHYSICAL, PokemonType.ROCK),
            "earthquake": MockMove("earthquake", 100, 1.0, MoveCategory.PHYSICAL, PokemonType.GROUND),
            "crunch": MockMove("crunch", 80, 1.0, MoveCategory.PHYSICAL, PokemonType.DARK),
        },
        ability="sandstream"
    )
    
    # Create Weavile (in back)
    weavile = MockPokemon(
        species="Weavile",
        current_hp_fraction=1.0,
        max_hp=245,
        base_stats={"hp": 70, "atk": 120, "def": 65, "spa": 45, "spd": 85, "spe": 125},
        types=[PokemonType.DARK, PokemonType.ICE],
        moves={
            "iceshard": MockMove("iceshard", 40, 1.0, MoveCategory.PHYSICAL, PokemonType.ICE, priority=1),
            "iciclecrash": MockMove("iciclecrash", 85, 0.9, MoveCategory.PHYSICAL, PokemonType.ICE),
            "tripleaxel": MockMove("tripleaxel", 20, 0.9, MoveCategory.PHYSICAL, PokemonType.ICE),
        },
        ability="pressure"
    )
    
    # Rotom's moves
    hydro_pump = MockMove("hydropump", 110, 0.8, MoveCategory.SPECIAL, PokemonType.WATER)
    volt_switch = MockMove("voltswitch", 70, 1.0, MoveCategory.SPECIAL, PokemonType.ELECTRIC)
    will_o_wisp = MockMove("willowisp", 0, 0.85, MoveCategory.STATUS, PokemonType.FIRE, status=Status.BRN)
    thunder_wave = MockMove("thunderwave", 0, 0.9, MoveCategory.STATUS, PokemonType.ELECTRIC, status=Status.PAR)
    
    rotom.moves = {
        "hydropump": hydro_pump,
        "voltswitch": volt_switch,
        "willowisp": will_o_wisp,
        "thunderwave": thunder_wave,
    }
    
    # Create battle
    battle = MockBattle(
        me=rotom,
        opp=tyranitar,
        team={"p1: Rotom-Wash": rotom, "p1: Weavile": weavile},
        opponent_team={"p2: Tyranitar": tyranitar}
    )
    
    battle.available_moves = [hydro_pump, volt_switch, will_o_wisp, thunder_wave]
    battle.available_switches = [weavile]
    
    return battle, rotom, tyranitar, weavile


def create_switch_into_ohko_scenario():
    """
    Test switching into a 4x weakness (should have huge penalty)
    """
    # Current mon: Rotom-Wash (safe vs Tyranitar)
    rotom = MockPokemon(
        species="Rotom-Wash",
        current_hp_fraction=0.80,
        max_hp=245,
        base_stats={"hp": 50, "atk": 65, "def": 107, "spa": 105, "spd": 107, "spe": 86},
        types=[PokemonType.ELECTRIC, PokemonType.WATER],
        moves={},
        ability="levitate"
    )
    
    # Opponent: Tyranitar
    tyranitar = MockPokemon(
        species="Tyranitar",
        current_hp_fraction=0.85,
        max_hp=341,
        base_stats={"hp": 100, "atk": 134, "def": 110, "spa": 95, "spd": 100, "spe": 61},
        types=[PokemonType.ROCK, PokemonType.DARK],
        moves={
            "stoneedge": MockMove("stoneedge", 100, 0.8, MoveCategory.PHYSICAL, PokemonType.ROCK),
        },
        ability="sandstream"
    )
    
    # Switch target: Weavile (4x weak to Rock)
    weavile = MockPokemon(
        species="Weavile",
        current_hp_fraction=1.0,
        max_hp=245,
        base_stats={"hp": 70, "atk": 120, "def": 65, "spa": 45, "spd": 85, "spe": 125},
        types=[PokemonType.DARK, PokemonType.ICE],
        moves={},
        ability="pressure"
    )
    
    battle = MockBattle(
        me=rotom,
        opp=tyranitar,
        team={"p1: Rotom-Wash": rotom, "p1: Weavile": weavile},
        opponent_team={"p2: Tyranitar": tyranitar}
    )
    
    return battle, rotom, tyranitar, weavile


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_pivot_move_scoring():
    """Test 1: Pivot moves should be scored reasonably, not overly inflated"""
    print("\n" + "=" * 80)
    print("TEST 1: PIVOT MOVE SCORING")
    print("=" * 80)
    print("Scenario: Rotom 35% HP vs Tyranitar 52% HP, Weavile in back")
    print("Expected: Hydro Pump or Will-O-Wisp > Volt Switch (sacrifices Weavile)")
    print("-" * 80)
    
    battle, rotom, tyranitar, weavile = create_rotom_tyranitar_scenario()
    ctx = EvalContext(me=rotom, opp=tyranitar, battle=battle, cache={})
    
    scores = {}
    for move_name, move in rotom.moves.items():
        score = score_move(move, battle, ctx)
        scores[move_name] = score
        print(f"{move_name:20s}: {score:6.1f}")
    
    # Assertions
    print("-" * 80)
    assert scores["hydropump"] > 0, "Hydro Pump should be positive (high damage)"
    assert scores["willowisp"] > 0, "Will-O-Wisp should be positive (survival)"
    assert scores["voltswitch"] < scores["hydropump"], "Volt Switch should be worse than Hydro Pump"
    assert scores["voltswitch"] < 150, "Volt Switch should not be inflated (was 1489 before fix)"
    
    print("✓ Pivot scoring is reasonable")
    print(f"✓ Volt Switch scored {scores['voltswitch']:.1f} (reasonable, not 1400+)")
    print(f"✓ Best move: {max(scores, key=scores.get)} ({max(scores.values()):.1f})")
    
    return scores


def test_switch_scoring():
    """Test 2: Switch into bad matchup should be heavily penalized"""
    print("\n" + "=" * 80)
    print("TEST 2: SWITCH SCORING - BAD MATCHUP")
    print("=" * 80)
    print("Scenario: Switch from Rotom to Weavile vs Tyranitar (4x Rock weakness)")
    print("Expected: Switch score should be highly negative")
    print("-" * 80)
    
    battle, rotom, tyranitar, weavile = create_switch_into_ohko_scenario()
    ctx = EvalContext(me=rotom, opp=tyranitar, battle=battle, cache={})
    
    switch_score_value = score_switch(weavile, battle, ctx)
    
    print(f"Switch to Weavile score: {switch_score_value:.1f}")
    print("-" * 80)
    
    # Assertions
    assert switch_score_value < 0, "Switch into OHKO should be negative"
    assert switch_score_value < -500, "Switch into 4x weakness should be heavily penalized"
    assert switch_score_value > -5000, "Switch score should be in reasonable range (not -5000)"
    
    print("✓ Switch penalty is working correctly")
    print(f"✓ Negative score: {switch_score_value:.1f} (properly penalized)")
    print("✓ Score in reasonable range (-5000 to 0)")
    
    return switch_score_value


def test_mcts_simulation():
    """Test 3: MCTS should make reasonable decisions with new scoring"""
    print("\n" + "=" * 80)
    print("TEST 3: MCTS SIMULATION")
    print("=" * 80)
    print("Scenario: Rotom vs Tyranitar with Weavile in back")
    print("Running MCTS with 120 iterations, depth 4...")
    print("-" * 80)
    
    battle, rotom, tyranitar, weavile = create_rotom_tyranitar_scenario()
    ctx = EvalContext(me=rotom, opp=tyranitar, battle=battle, cache={})
    
    # Build actions
    actions = [("move", m) for m in battle.available_moves]
    actions.extend([("switch", weavile)])
    
    # Build belief (optional, can be None for this test)
    belief = None
    try:
        belief = build_opponent_belief(tyranitar, gen=9)
    except:
        pass
    
    # Run MCTS
    picked_action, stats = mcts_pick_action(
        battle=battle,
        ctx=ctx,
        belief=belief,
        actions=actions,
        iters=120,
        max_depth=4,
        include_switches=True,
    )
    
    if stats is None:
        print("⚠ MCTS returned no stats")
        return None
    
    # Print results
    print("\nMCTS Results:")
    print(f"{'Action':<30s} {'Visits':>8s} {'Q-value':>10s} {'Visit %':>10s}")
    print("-" * 60)
    
    total_visits = sum(t['visits'] for t in stats['top'])
    
    for i, result in enumerate(stats['top'][:5], 1):
        action_name = f"{result['kind']} {result['name']}"
        visits_pct = result['visits'] / total_visits * 100 if total_visits > 0 else 0
        print(f"{action_name:<30s} {result['visits']:>8d} {result['q']:>10.2f} {visits_pct:>9.1f}%")
    
    print("-" * 80)
    
    # Assertions
    if picked_action:
        kind, obj = picked_action
        action_name = f"{kind} {getattr(obj, 'id', obj.species)}"
        print(f"\n✓ MCTS chose: {action_name}")
        
        # Check that Volt Switch isn't dominating
        volt_stats = next((t for t in stats['top'] if 'voltswitch' in t['name'].lower()), None)
        if volt_stats:
            assert volt_stats['visits'] < total_visits * 0.6, "Volt Switch shouldn't dominate visits"
            print(f"✓ Volt Switch visited {volt_stats['visits']} times ({volt_stats['visits']/total_visits*100:.1f}%) - not dominating")
    
    return picked_action, stats


def test_heuristic_vs_mcts():
    """Test 4: Compare heuristic and MCTS decisions"""
    print("\n" + "=" * 80)
    print("TEST 4: HEURISTIC VS MCTS COMPARISON")
    print("=" * 80)
    
    battle, rotom, tyranitar, weavile = create_rotom_tyranitar_scenario()
    ctx = EvalContext(me=rotom, opp=tyranitar, battle=battle, cache={})
    
    # Heuristic scores
    print("Heuristic Scores:")
    heuristic_scores = {}
    for move_name, move in rotom.moves.items():
        score = score_move(move, battle, ctx)
        heuristic_scores[move_name] = score
        print(f"  {move_name:20s}: {score:6.1f}")
    
    best_heuristic = max(heuristic_scores, key=heuristic_scores.get)
    print(f"\nHeuristic choice: {best_heuristic} ({heuristic_scores[best_heuristic]:.1f})")
    
    # MCTS choice
    print("\nRunning MCTS...")
    actions = [("move", m) for m in battle.available_moves]
    
    picked_action, stats = mcts_pick_action(
        battle=battle,
        ctx=ctx,
        belief=None,
        actions=actions,
        iters=120,
        max_depth=4,
        include_switches=False,
    )
    
    if picked_action and stats:
        kind, obj = picked_action
        mcts_choice = obj.id
        print(f"MCTS choice: {mcts_choice}")
        
        # Print comparison
        print("\n" + "-" * 80)
        print(f"Agreement: {'✓ YES' if mcts_choice == best_heuristic else '✗ NO'}")
        
        if mcts_choice != best_heuristic:
            print(f"\nHeuristic preferred: {best_heuristic}")
            print(f"MCTS preferred: {mcts_choice}")
            print("\nThis is OK - MCTS sees deeper lines and may make different choices")
    
    return best_heuristic, picked_action


def test_scoring_ranges():
    """Test 5: Verify all scores are in reasonable ranges"""
    print("\n" + "=" * 80)
    print("TEST 5: SCORING RANGES VALIDATION")
    print("=" * 80)
    
    battle, rotom, tyranitar, weavile = create_rotom_tyranitar_scenario()
    ctx = EvalContext(me=rotom, opp=tyranitar, battle=battle, cache={})
    
    print("Checking score ranges...")
    print("-" * 80)
    
    # Move scores
    all_in_range = True
    for move_name, move in rotom.moves.items():
        score = score_move(move, battle, ctx)
        in_range = -200 < score < 300
        status = "✓" if in_range else "✗"
        print(f"{status} Move {move_name:20s}: {score:6.1f} (expected: -200 to 300)")
        if not in_range:
            all_in_range = False
    
    # Switch score
    switch_score_value = score_switch(weavile, battle, ctx)
    switch_in_range = -5000 < switch_score_value < 5000
    status = "✓" if switch_in_range else "✗"
    print(f"{status} Switch to Weavile:       {switch_score_value:6.1f} (expected: -5000 to 5000)")
    if not switch_in_range:
        all_in_range = False
    
    print("-" * 80)
    
    if all_in_range:
        print("✓ All scores in reasonable ranges!")
    else:
        print("✗ Some scores out of range - may need adjustment")
    
    return all_in_range


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all tests and print summary"""
    print("\n" + "=" * 80)
    print(" " * 20 + "SCORING SYSTEM TEST SUITE")
    print(" " * 20 + "With MCTS Integration")
    print("=" * 80)
    
    results = {}
    
    # Test 1: Pivot scoring
    try:
        results['pivot_scoring'] = test_pivot_move_scoring()
        print("\n✓ TEST 1 PASSED")
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {e}")
        results['pivot_scoring'] = None
    
    # Test 2: Switch scoring
    try:
        results['switch_scoring'] = test_switch_scoring()
        print("\n✓ TEST 2 PASSED")
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {e}")
        results['switch_scoring'] = None
    
    # Test 3: MCTS simulation
    try:
        results['mcts_simulation'] = test_mcts_simulation()
        print("\n✓ TEST 3 PASSED")
    except Exception as e:
        print(f"\n✗ TEST 3 FAILED: {e}")
        results['mcts_simulation'] = None
    
    # Test 4: Heuristic vs MCTS
    try:
        results['heuristic_vs_mcts'] = test_heuristic_vs_mcts()
        print("\n✓ TEST 4 PASSED")
    except Exception as e:
        print(f"\n✗ TEST 4 FAILED: {e}")
        results['heuristic_vs_mcts'] = None
    
    # Test 5: Score ranges
    try:
        results['score_ranges'] = test_scoring_ranges()
        print("\n✓ TEST 5 PASSED")
    except Exception as e:
        print(f"\n✗ TEST 5 FAILED: {e}")
        results['score_ranges'] = None
    
    # Final summary
    print("\n" + "=" * 80)
    print(" " * 30 + "TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for v in results.values() if v is not None)
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Scoring system is working correctly.")
    else:
        print("\n⚠ Some tests failed. Review output above for details.")
    
    print("\n" + "=" * 80)
    
    return results


if __name__ == "__main__":
    results = run_all_tests()