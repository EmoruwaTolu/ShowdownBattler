import sys
import os
from typing import Any, Dict, List
from unittest.mock import Mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poke_env.battle import MoveCategory, PokemonType, Status, Pokemon, Move, Battle
from poke_env.data import GenData
from bot.model.ctx import EvalContext
from bot.mcts.search import search, MCTSConfig, format_tree
from bot.scoring.move_score import score_move
from bot.scoring.switch_score import score_switch
from bot.scoring.damage_score import estimate_damage_fraction

def create_mock_pokemon(
    species: str,
    identifier: str,  # e.g., "p1: Garchomp"
    hp_frac: float = 1.0,
    level: int = 50,
    types: tuple = (PokemonType.NORMAL,),
    stats: Dict[str, int] = None,
    boosts: Dict[str, int] = None,
    ability: str = None,
    item: str = None,
) -> Any:
    """
    Create a Pokemon mock that works with calculate_damage.
    
    Key requirements from calculate_damage:
    - Must have stats as dict with int/float values (not None)
    - Must have boosts dict
    - Must have types (type_1, type_2, types list)
    - Must have level, ability, item
    - Must have current_hp_fraction
    - Must be in battle.team or battle.opponent_team
    """
    mon = Mock(spec=Pokemon)
    
    # Identity
    mon.species = species
    mon.base_species = species
    
    if stats is None:
        stats = {
            'hp': 150,
            'atk': 100,
            'def': 100,
            'spa': 100,
            'spd': 100,
            'spe': 100,
        }
    mon.stats = stats
    
    mon.base_stats = stats.copy()  # Use same values for simplicity
    
    # Boosts (REQUIRED)
    if boosts is None:
        boosts = {'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0, 'accuracy': 0, 'evasion': 0}
    mon.boosts = boosts
    
    # HP
    mon.current_hp_fraction = hp_frac
    mon.max_hp = stats.get('hp', 150)
    mon.current_hp = int(mon.max_hp * hp_frac)  # Add this for damage calc!
    
    # Level
    mon.level = level
    
    # Types
    if len(types) >= 1:
        mon.type_1 = types[0]
        mon.type_2 = types[1] if len(types) > 1 else None
        mon.types = list(types)
        mon.original_types = list(types)
    else:
        mon.type_1 = PokemonType.NORMAL
        mon.type_2 = None
        mon.types = [PokemonType.NORMAL]
        mon.original_types = [PokemonType.NORMAL]
    
    # Ability and item
    mon.ability = ability
    mon.item = item
    mon.possible_abilities = [ability] if ability else []
    
    # Status and effects
    mon.status = None
    mon.effects = {}
    
    # Weight (for some move calcs)
    mon.weight = 100.0
    
    # Tera
    mon.is_terastallized = False
    mon.tera_type = None
    
    # Gender
    mon.gender = None
    
    # Fainted
    mon.fainted = False
    
    # Moves
    mon.moves = {}
    
    # Add identifier method (REQUIRED for calculate_damage)
    # The damage calculator calls pokemon.identifier(role) where role is "p1" or "p2"
    # It expects this to return something like "p1: Garchomp"
    mon._identifier_string = identifier
    
    def get_identifier(role=None):
        # If called with no arguments or with a role, return the identifier
        return mon._identifier_string
    
    mon.identifier = get_identifier
    
    # Add _data attribute for eviolite check
    mon._data = Mock()
    mon._data.pokedex = {species: {"evos": []}}
    
    return mon


def create_mock_move(
    move_id: str,
    category: MoveCategory,
    move_type: PokemonType,
    base_power: int = 0,
    accuracy: float = 1.0,
    priority: int = 0,
    crit_ratio: int = 0,
    status: Status = None,
    secondary: list = None, 
) -> Any:
    """
    Create a Move mock that works with calculate_damage.
    
    Key requirements:
    - Must have entry dict with flags, basePower, type, category
    - Must have n_hit tuple
    - Must have all damage-related attributes
    - Must have current_pp and max_pp as integers
    - Must have crit_ratio as integer
    """
    move = Mock(spec=Move)
    
    # Basic attributes
    move.id = move_id
    move.category = category
    move.type = move_type
    move.base_power = base_power
    move.accuracy = accuracy
    move.priority = priority
    move.crit_ratio = crit_ratio
    move.status = status
    move.secondary = secondary if secondary is not None else []
    
    # PP (REQUIRED)
    move.current_pp = 16
    move.max_pp = 16
    
    # Entry dict (REQUIRED for calculate_damage)
    move.entry = {
        'flags': {},  # e.g., {'contact': 1, 'punch': 1}
        'basePower': base_power,
        'type': move_type.name,
        'category': category.name,
        'critRatio': crit_ratio,  # Also add to entry dict
    }
    
    # Multi-hit (REQUIRED)
    move.n_hit = (1, 1)  # Single hit
    
    # Other attributes
    move.breaks_protect = False
    move.ignore_defensive = False
    move.target = None
    move.recoil = None
    move.secondary = None
    move.is_stellar_first_use = False
    move.flags = {}
    
    return move


def create_mock_battle(
    active_identifier: str,
    active: Any,
    opponent_identifier: str,
    opponent: Any,
    team: Dict[str, Any],
    opponent_team: Dict[str, Any],
) -> Any:
    """
    Create a Battle mock that works with calculate_damage.
    
    Key requirements:
    - Must have player_role and opponent_role
    - Must have team and opponent_team dicts
    - Must have get_pokemon() method that returns Pokemon from teams
    - Must have all_active_pokemons list
    - Must have fields, weather, side_conditions
    """
    battle = Mock(spec=Battle)
    
    # Teams (REQUIRED calculate_damage uses these)
    battle.team = team
    battle.opponent_team = opponent_team
    
    # Active Pokemon
    battle.active_pokemon = active
    battle.opponent_active_pokemon = opponent
    
    # Also set these alternate names (damage calc might use different names)
    battle.player_active_pokemon = active
    
    # Set active role (damage calc might check this)
    battle.player_username = "Player1"
    battle.opponent_username = "Player2"
    
    # Roles (REQUIRED)
    battle.player_role = "p1"
    battle.opponent_role = "p2"
    
    # All active (REQUIRED)
    # Make sure both Pokemon are actually in the list and not None
    battle.all_active_pokemons = [p for p in [active, opponent] if p is not None]
    
    # Fields, weather, conditions (REQUIRED)
    battle.fields = {}
    battle.weather = {}
    battle.side_conditions = {}
    battle.opponent_side_conditions = {}
    
    # Available moves/switches
    battle.available_moves = []
    battle.available_switches = []
    
    # Format
    battle.format = Mock()
    battle.format.gen = 9
    
    # get_pokemon method (REQUIRED for calculate_damage)
    def get_pokemon(identifier: str):
        """Look up Pokemon by identifier - with detailed error logging"""
        if identifier in team:
            return team[identifier]
        elif identifier in opponent_team:
            return opponent_team[identifier]
        # If not found, this is a problem - log it
        print(f"WARNING: get_pokemon() called with unknown identifier: '{identifier}'")
        print(f"  Available in team: {list(team.keys())}")
        print(f"  Available in opponent_team: {list(opponent_team.keys())}")
        return None
    
    battle.get_pokemon = get_pokemon
    
    # is_grounded method (REQUIRED for some move calcs)
    def is_grounded(pokemon):
        """Check if Pokemon is grounded"""
        if pokemon is None:
            return True
        # Simple check: Flying types aren't grounded
        types = getattr(pokemon, 'types', [])
        if PokemonType.FLYING in types:
            return False
        if getattr(pokemon, 'ability', None) == "levitate":
            return False
        return True
    
    battle.is_grounded = is_grounded
    
    return battle

def create_garchomp_rotom_scenario():
    """
    Garchomp vs Rotom-Wash - test real damage calculations
    """
    garchomp = create_mock_pokemon(
        species="Garchomp",
        identifier="p1: Garchomp",
        hp_frac=0.85,
        level=50,
        types=(PokemonType.GROUND, PokemonType.DRAGON),
        stats={
            'hp': 183,
            'atk': 182,  # High attack
            'def': 115,
            'spa': 100,
            'spd': 105,
            'spe': 169,
        },
        ability="roughskin",
    )
    
    # Create Rotom-Wash with proper stats
    rotom = create_mock_pokemon(
        species="Rotom-Wash",
        identifier="p2: Rotom",
        hp_frac=0.60,
        level=50,
        types=(PokemonType.ELECTRIC, PokemonType.WATER),
        stats={
            'hp': 127,
            'atk': 85,
            'def': 147,  # High defense
            'spa': 125,
            'spd': 147,
            'spe': 126,
        },
        ability="levitate",
    )

    earthquake = create_mock_move(
        "earthquake",
        MoveCategory.PHYSICAL,
        PokemonType.GROUND,
        base_power=100,
        accuracy=1.0,
        crit_ratio=0,  # Normal crit rate
    )
    earthquake.entry['flags'] = {}
    
    outrage = create_mock_move(
        "outrage",
        MoveCategory.PHYSICAL,
        PokemonType.DRAGON,
        base_power=120,
        accuracy=1.0,
        crit_ratio=0,  # Normal crit rate
    )
    
    stoneedge = create_mock_move(
        "stoneedge",
        MoveCategory.PHYSICAL,
        PokemonType.ROCK,
        base_power=100,
        accuracy=0.8,
        crit_ratio=2,  # High crit rate (50%)
    )
    
    garchomp.moves = {
        "earthquake": earthquake,
        "outrage": outrage,
        "stoneedge": stoneedge,
    }
    
    hydropump = create_mock_move(
        "hydropump",
        MoveCategory.SPECIAL,
        PokemonType.WATER,
        base_power=110,
        accuracy=0.8,
        crit_ratio=0,  # Normal crit rate
    )
    
    rotom.moves = {"hydropump": hydropump}
    
    team = {"p1: Garchomp": garchomp}
    opp_team = {"p2: Rotom": rotom}
    
    battle = create_mock_battle(
        active_identifier="p1: Garchomp",
        active=garchomp,
        opponent_identifier="p2: Rotom",
        opponent=rotom,
        team=team,
        opponent_team=opp_team
    )
    
    battle.available_moves = [earthquake, outrage, stoneedge]
    
    return battle, garchomp, rotom

def test_1_damage_calculator():
    """
    Test that calculate_damage works with our mocks.
    """
    print("\n" + "=" * 80)
    print("TEST 1: DAMAGE CALCULATOR VERIFICATION")
    print("=" * 80)
    
    battle, garchomp, rotom = create_garchomp_rotom_scenario()
    
    print("Testing damage calculations:")
    print()
    
    for move_name, move in garchomp.moves.items():
        try:
            dmg_frac = estimate_damage_fraction(move, garchomp, rotom, battle)
            print(f"  {move_name:15s}: {dmg_frac:.3f} ({dmg_frac*100:.1f}% damage)")
        except Exception as e:
            print(f"  {move_name:15s}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print()
    print("Damage calculator works with mocks!")
    return True

def test_2_real_heuristics():
    """
    Test your actual scoring functions.
    """
    print("\n" + "=" * 80)
    print("TEST 2: REAL HEURISTICS")
    print("=" * 80)
    
    battle, garchomp, rotom = create_garchomp_rotom_scenario()
    ctx = EvalContext(me=garchomp, opp=rotom, battle=battle, cache={})
    
    print("Testing your actual move_score function:")
    print()
    
    scores = {}
    for move_name, move in garchomp.moves.items():
        try:
            score = score_move(move, battle, ctx)
            scores[move_name] = score
            print(f"  {move_name:15s}: {score:7.2f}")
        except Exception as e:
            print(f"  {move_name:15s}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print()
    
    if scores:
        best_move = max(scores, key=scores.get)
        print(f"Best move: {best_move} (score: {scores[best_move]:.2f})")
        print("Real heuristics work!")
        return True
    else:
        print("No scores computed")
        return False

def test_3_mcts_real_heuristics():
    """
    Test MCTS using your real scoring functions.
    """
    print("\n" + "=" * 80)
    print("TEST 3: MCTS WITH REAL HEURISTICS")
    print("=" * 80)
    
    battle, garchomp, rotom = create_garchomp_rotom_scenario()
    
    ctx_me = EvalContext(me=garchomp, opp=rotom, battle=battle, cache={})
    ctx_opp = EvalContext(me=rotom, opp=garchomp, battle=battle, cache={})
    
    cfg = MCTSConfig(
        num_simulations=100,
        seed=42,
        use_hybrid_expansion=False,
    )
    
    print("Running MCTS (100 simulations)...")
    print()
    
    try:
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
        
        kind, obj = picked
        print(f"Picked: {kind} {getattr(obj, 'id', 'unknown')}")
        print()
        
        print("Top 3 actions:")
        for i, action_stats in enumerate(stats['top'][:3], 1):
            print(f"  {i}. {action_stats['kind']} {action_stats['name']}")
            print(f"     Visits: {action_stats['visits']}, Q: {action_stats['q']:+.3f}")
        
        print()
        print("MCTS works with real heuristics!")
        return True
        
    except Exception as e:
        print(f"MCTS failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_4_hybrid_expansion():
    """
    Test hybrid expansion with real heuristics.
    """
    print("\n" + "=" * 80)
    print("TEST 4: HYBRID EXPANSION WITH REAL HEURISTICS")
    print("=" * 80)
    
    garchomp = create_mock_pokemon(
        species="Garchomp",
        identifier="p1: Garchomp",
        hp_frac=0.85,
        level=50,
        types=(PokemonType.GROUND, PokemonType.DRAGON),
        stats={'hp': 183, 'atk': 182, 'def': 115, 'spa': 100, 'spd': 105, 'spe': 169},
    )
    
    moltres = create_mock_pokemon(
        species="Moltres",
        identifier="p2: Moltres",
        hp_frac=0.65,
        level=50,
        types=(PokemonType.FIRE, PokemonType.FLYING),
        stats={'hp': 165, 'atk': 120, 'def': 110, 'spa': 145, 'spd': 105, 'spe': 110},
    )
    
    stoneedge = create_mock_move("stoneedge", MoveCategory.PHYSICAL, PokemonType.ROCK, 100, 0.8, crit_ratio=2)
    garchomp.moves = {"stoneedge": stoneedge}
    
    flamethrower = create_mock_move("flamethrower", MoveCategory.SPECIAL, PokemonType.FIRE, 90, 1.0, crit_ratio=0)
    moltres.moves = {"flamethrower": flamethrower}
    
    team = {"p1: Garchomp": garchomp}
    opp_team = {"p2: Moltres": moltres}
    
    battle = create_mock_battle(
        "p1: Garchomp", garchomp,
        "p2: Moltres", moltres,
        team, opp_team
    )
    battle.available_moves = [stoneedge]
    
    ctx_me = EvalContext(me=garchomp, opp=moltres, battle=battle, cache={})
    ctx_opp = EvalContext(me=moltres, opp=garchomp, battle=battle, cache={})
    
    cfg = MCTSConfig(
        num_simulations=100,
        seed=42,
        use_hybrid_expansion=True,
        branch_low_accuracy=True,
        low_accuracy_threshold=0.85,
        min_branch_probability=0.01,
    )
    
    print("Stone Edge (80% acc, 4x vs Fire/Flying)")
    
    try:
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
        
        root = stats['root']
        
        print(f"\nBranches: {len(root.children)}")
        print()
        
        for action, child in root.children.items():
            outcome = action[2] if len(action) > 2 else "standard"
            print(f"  {outcome:12s}: N={child.N:3d}, Q={child.Q:+.3f}, P={child.prior:.4f}")
        
        print()
        print("Hybrid expansion works with real heuristics!")
        return True
        
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "REAL HEURISTICS + HYBRID MCTS INTEGRATION" + " " * 21 + "║")
    print("╚" + "=" * 78 + "╝")
    
    tests = [
        ("Damage Calculator", test_1_damage_calculator),
        ("Real Heuristics", test_2_real_heuristics),
        ("MCTS + Real Heuristics", test_3_mcts_real_heuristics),
        ("Hybrid Expansion", test_4_hybrid_expansion),
    ]
    
    results = {}
    
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n{name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "Yes" if result else "No"
        print(f"  {status} {name}")
    
    print()
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\nSUCCESS! Your real heuristics work with hybrid MCTS!")
    
    return results


if __name__ == "__main__":
    run_all_tests()