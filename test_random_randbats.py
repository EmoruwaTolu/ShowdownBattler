import sys
import os
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List, Optional
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

def load_randbats_data():
    possible_paths = [
        "gen9randombattle.json",
        os.path.join(os.path.dirname(__file__), "..", "data", "randbats", "gen9randombattle.json"),
    ]
    
    for json_path in possible_paths:
        if os.path.exists(json_path):
            print(f"  Found at: {json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    # If not found, show where we looked
    raise FileNotFoundError(
        f"Could not find gen9randombattle.json. Searched in:\n" +
        "\n".join(f"  - {p}" for p in possible_paths)
    )


def get_pokemon_info(species_name: str, gen_data) -> Dict:
    """Get Pokemon info from GenData pokedex."""
    species_id = to_id_str(species_name)
    return gen_data.pokedex.get(species_id, {})


def get_move_info(move_name: str, gen_data) -> Dict:
    """Get move info from GenData moves."""
    move_id = to_id_str(move_name)
    return gen_data.moves.get(move_id, {})


def get_pokemon_types(species_name: str, gen_data) -> tuple:
    """Get types for a Pokemon from GenData."""
    pkmn_info = get_pokemon_info(species_name, gen_data)
    
    if not pkmn_info:
        return (PokemonType.NORMAL,)
    
    types = []
    type_1_str = pkmn_info.get("types", ["Normal"])[0]
    types.append(PokemonType.from_name(type_1_str))
    
    type_list = pkmn_info.get("types", [])
    if len(type_list) > 1:
        type_2_str = type_list[1]
        types.append(PokemonType.from_name(type_2_str))
    
    return tuple(types)


def get_pokemon_base_stats(species_name: str, gen_data) -> Dict[str, int]:
    """Get base stats for a Pokemon from GenData."""
    pkmn_info = get_pokemon_info(species_name, gen_data)
    
    if not pkmn_info or "baseStats" not in pkmn_info:
        return {
            'hp': 80,
            'atk': 80,
            'def': 80,
            'spa': 80,
            'spd': 80,
            'spe': 80,
        }
    
    return pkmn_info["baseStats"]


def calculate_stat(base: int, level: int, ev: int = 85) -> int:
    """Calculate actual stat from base stat."""
    return int(((2 * base + 31 + ev // 4) * level // 100 + 5))


def calculate_hp(base: int, level: int, ev: int = 85) -> int:
    """Calculate HP stat from base HP."""
    return int((2 * base + 31 + ev // 4) * level // 100 + level + 10)


def get_move_type(move_name: str, gen_data) -> PokemonType:
    """Get type for a move from GenData."""
    move_info = get_move_info(move_name, gen_data)
    
    if not move_info:
        return PokemonType.NORMAL
    
    type_str = move_info.get("type", "Normal")
    try:
        return PokemonType.from_name(type_str)
    except:
        return PokemonType.NORMAL


def get_move_category(move_name: str, gen_data) -> MoveCategory:
    """Get category for a move from GenData."""
    move_info = get_move_info(move_name, gen_data)
    
    if not move_info:
        return MoveCategory.SPECIAL
    
    category_str = move_info.get("category", "Special")
    
    if category_str == "Physical":
        return MoveCategory.PHYSICAL
    elif category_str == "Special":
        return MoveCategory.SPECIAL
    else:
        return MoveCategory.STATUS


def get_move_power(move_name: str, gen_data) -> int:
    """Get base power for a move from GenData."""
    move_info = get_move_info(move_name, gen_data)
    
    if not move_info:
        return 0
    
    return move_info.get("basePower", 0)


def get_move_accuracy(move_name: str, gen_data) -> float:
    """Get accuracy for a move from GenData."""
    move_info = get_move_info(move_name, gen_data)
    
    if not move_info:
        return 1.0
    
    accuracy = move_info.get("accuracy", True)
    
    if accuracy is True:
        return 1.0
    
    return float(accuracy) / 100.0


def get_move_crit_ratio(move_name: str, gen_data) -> int:
    """Get crit ratio for a move from GenData."""
    move_info = get_move_info(move_name, gen_data)
    
    if not move_info:
        return 0
    
    # critRatio field: 0 (normal), 1 (Focus Energy), 2 (high crit moves), 3+ (always crit)
    return move_info.get("critRatio", 0)

def get_move_status(move_name: str, gen_data) -> Optional[Status]:
    """Get status effect for a move from GenData."""
    move_info = get_move_info(move_name, gen_data)
    
    if not move_info:
        return None
    
    status_str = move_info.get("status", None)
    if not status_str:
        return None
    
    try:
        return Status[status_str.upper()]
    except (KeyError, AttributeError):
        return None
    
def get_move_secondary(move_name: str, gen_data) -> Optional[list]:
    """Get secondary effects for a move."""
    move_info = get_move_info(move_name, gen_data)
    if not move_info:
        return None
    
    secondary = move_info.get("secondary", None)
    if not secondary:
        return None
    
    if not isinstance(secondary, list):
        secondary = [secondary]
    
    return secondary

def get_move_boosts(move_name: str, gen_data) -> Optional[Dict[str, int]]:
    """Get guaranteed stat boosts (e.g., Nasty Plot)."""
    move_info = get_move_info(move_name, gen_data)
    if not move_info:
        return None
    return move_info.get('boosts', None)


def get_move_self(move_name: str, gen_data) -> Optional[Dict]:
    """Get self-inflicted effects (e.g., Draco Meteor drops SpA)."""
    move_info = get_move_info(move_name, gen_data)
    if not move_info:
        return None
    return move_info.get('self', None)

def create_pokemon_from_randbats(species_name: str, data: Dict, gen_data, role_name: str = None) -> Any:
    """
    Create a mock Pokemon from RandBats data using GenData for accurate info.
    
    If role_name is specified, use that role. Otherwise pick first role.
    """
    species_id = to_id_str(species_name)
    level = data.get("level", 50)
    
    # Get role data
    roles = data.get("roles", {})
    if not roles:
        raise ValueError(f"No roles found for {species_name}")
    
    if role_name and role_name in roles:
        role = roles[role_name]
    else:
        role_name = random.choice(list(roles.keys())) # randomly choosing one of the roles from the json
        role = roles[role_name]
    
    # Get moves from role
    move_names = role.get("moves", [])
    if not move_names:
        raise ValueError(f"No moves found for {species_name} - {role_name}")
    
    # Get ability and item (but we won't use them since heuristics don't account for them)
    abilities = role.get("abilities", data.get("abilities", []))
    ability = abilities[0] if abilities else None
    
    items = role.get("items", data.get("items", []))
    item = items[0] if items else None
    
    # Get types from GenData
    types = get_pokemon_types(species_name, gen_data)
    
    # Get base stats from GenData
    base_stats = get_pokemon_base_stats(species_name, gen_data)
    
    # Calculate actual stats from base stats
    stats = {
        'hp': calculate_hp(base_stats['hp'], level),
        'atk': calculate_stat(base_stats['atk'], level),
        'def': calculate_stat(base_stats['def'], level),
        'spa': calculate_stat(base_stats['spa'], level),
        'spd': calculate_stat(base_stats['spd'], level),
        'spe': calculate_stat(base_stats['spe'], level),
    }
    
    # Create Pokemon w/o ability/item (heuristics don't use them yet)
    # Don't set identifier here - caller will set it based on which team
    mon = create_mock_pokemon(
        species=species_name,
        identifier="",  # Will be overridden by caller
        hp_frac=1.0,
        level=level,
        types=types,
        stats=stats,
        ability=None,  # Don't set ability
        item=None,     # Don't set item
    )
    
    # Create moves (pick up to 4 random moves from the role)
    num_moves = min(4, len(move_names))
    selected_moves = random.sample(move_names, num_moves)
    
    moves_dict = {}
    for move_name in selected_moves:
        move_id = to_id_str(move_name)
        move_type = get_move_type(move_name, gen_data)
        move_cat = get_move_category(move_name, gen_data)
        move_power = get_move_power(move_name, gen_data)
        move_acc = get_move_accuracy(move_name, gen_data)
        move_crit = get_move_crit_ratio(move_name, gen_data)
        move_status = get_move_status(move_name, gen_data)
        move_secondary = get_move_secondary(move_name, gen_data)
        move_boosts = get_move_boosts(move_name, gen_data)
        move_self = get_move_self(move_name, gen_data)  
        
        move = create_mock_move(
            move_id=move_id,
            category=move_cat,
            move_type=move_type,
            base_power=move_power,
            accuracy=move_acc,
            crit_ratio=move_crit,
            status=move_status,
            secondary=move_secondary,
            boosts=move_boosts,   
            self_effect=move_self
        )
        
        moves_dict[move_id] = move
    
    mon.moves = moves_dict

    print(f"\n=== DIAGNOSTIC: {species_name} ===")
    print(f"JSON level: {level}")
    print(f"mon.level: {mon.level}")
    print(f"Base HP: {base_stats['hp']}")
    print(f"mon.stats['hp']: {mon.stats['hp']}")
    print(f"Expected at L{level}: {calculate_hp(base_stats['hp'], level)}")
    print(f"Expected at L50: {calculate_hp(base_stats['hp'], 50)}")
    print(f"Moves:")
    for move_id, move in mon.moves.items():  
        print(f"  - {move.id}: {move.base_power} BP, {move.category.name}, {move.type.name}")
        
    return mon, role_name, selected_moves, ability, item

def test_random_randbats_matchup():
    """
    Pick two random RandBats Pokemon and show the MCTS tree.
    """
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "RANDOM RANDBATS MATCHUP" + " " * 35 + "║")
    print("╚" + "=" * 78 + "╝")
    
    gen_data = GenData.from_gen(9)
    
    randbats_data = load_randbats_data()
    
    all_species = list(randbats_data.keys())
    print(f"Found {len(all_species)} species in RandBats data")
    
    # Pick two random Pokemon
    random.seed()
    player_species, opponent_species = random.sample(all_species, 2)
    
    print("\n" + "=" * 80)
    print("MATCHUP SELECTION")
    print("=" * 80)
    print(f"Player: {player_species}")
    print(f"Opponent: {opponent_species}")
    
    # Create Pokemon
    player_mon, player_role, player_moves, player_ability, player_item = create_pokemon_from_randbats(
        player_species, randbats_data[player_species], gen_data
    )
    
    opponent_mon, opp_role, opp_moves, opp_ability, opp_item = create_pokemon_from_randbats(
        opponent_species, randbats_data[opponent_species], gen_data
    )
    
    # Species is already set correctly in create_mock_pokemon but we need to set the correct identifiers to match the team dict keys
    player_mon._identifier_string = f"p1: {player_species}"
    opponent_mon._identifier_string = f"p2: {opponent_species}"
    
    print(f"\n{player_species} ({player_role}):")
    print(f"Ability: {player_ability} (not used in heuristics)")
    print(f"Item: {player_item} (not used in heuristics)")
    print(f"Moves: {', '.join(player_moves)}")
    print(f"Types: {', '.join(str(t).split('.')[-1] for t in player_mon.types)}")
    
    print(f"\n{opponent_species} ({opp_role}):")
    print(f"Ability: {opp_ability} (not used in heuristics)")
    print(f"Item: {opp_item} (not used in heuristics)")
    print(f"Moves: {', '.join(opp_moves)}")
    print(f"Types: {', '.join(str(t).split('.')[-1] for t in opponent_mon.types)}")
    
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
    
    # Show heuristic scores first
    print("\n" + "=" * 80)
    print("HEURISTIC SCORES (before MCTS)")
    print("=" * 80)
    
    print("\nDEBUG: Checking Pokemon setup")
    print(f"player_mon object id: {id(player_mon)}")
    print(f"team[key] object id: {id(team[f'p1: {player_species}'])}")
    print(f"Same object? {player_mon is team[f'p1: {player_species}']}")
    print(f"player_mon.species: {player_mon.species}")
    
    print(f"\nopponent_mon object id: {id(opponent_mon)}")
    print(f"opp_team[key] object id: {id(opp_team[f'p2: {opponent_species}'])}")
    print(f"Same object? {opponent_mon is opp_team[f'p2: {opponent_species}']}")
    print(f"opponent_mon.species: {opponent_mon.species}")
    
    # Test identifier lookup directly
    from bot.scoring.damage_score import _get_pokemon_identifier
    player_id = _get_pokemon_identifier(player_mon, battle)
    opp_id = _get_pokemon_identifier(opponent_mon, battle)
    print(f"\n  player identifier lookup: {player_id}")
    print(f"opponent identifier lookup: {opp_id}")
    
    print("\nHeuristic scores:")
    for move_id, move in player_mon.moves.items():
        try:
            # Test damage calculation directly
            from bot.scoring.damage_score import estimate_damage_fraction, _get_pokemon_identifier
            
            # Verify identifiers
            me_id = _get_pokemon_identifier(player_mon, battle)
            opp_id = _get_pokemon_identifier(opponent_mon, battle)
            
            print(f"\n{move_id}:")
            print(f"me_id: {me_id}, opp_id: {opp_id}")
            
            # Try damage calc with detailed error catching
            try:
                from poke_env.calc.damage_calc_gen9 import calculate_damage
                min_dmg, max_dmg = calculate_damage(
                    attacker_identifier=me_id,
                    defender_identifier=opp_id,
                    move=move,
                    battle=battle,
                    is_critical=False,
                )
                print(f"calculate_damage returned: min={min_dmg}, max={max_dmg}")
                dmg = (min_dmg + max_dmg) / 2.0 / opponent_mon.max_hp
            except Exception as e:
                print(f"calculate_damage FAILED: {type(e).__name__}: {e}")
                dmg = 0.25
            
            # Get full score
            score = score_move(move, battle, ctx_me)
            print(f"    score: {score:.2f}, dmg_frac: {dmg:.3f}")
            
        except Exception as e:
            print(f"  {move_id:20s}: ERROR - {e}")
    
    # Run MCTS
    print("\n" + "=" * 80)
    print("RUNNING MCTS")
    print("=" * 80)
    print("Simulations: 200")
    print("Max Depth: 4")
    print("Hybrid Expansion: Disabled")
    
    cfg = MCTSConfig(
        num_simulations=200,
        max_depth=4,
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
            opp_tau=0.0001,
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
        
        print("\nTop 3 Actions by Visits:")
        for i, action_stats in enumerate(stats['top'][:3], 1):
            print(f"  {i}. {action_stats['kind']} {action_stats['name']}")
            print(f"     Visits: {action_stats['visits']:3d}, Q: {action_stats['q']:+.3f}, Prior: {action_stats['prior']:.4f}")
        
        # Visualize tree
        root = stats['root']
        visualize_tree_depth(root, max_depth=2)
        
        print("\n" + "=" * 80)
        print("ANALYSIS")
        print("=" * 80)
        print(f"Total simulations: {root.N}")
        print(f"Root Q-value: {root.Q:+.3f}")
        print(f"Number of root actions: {len(root.children)}")
        
        # Show visit distribution
        print("\nVisit Distribution:")
        total_visits = sum(child.N for child in root.children.values())
        for action, child in sorted(root.children.items(), key=lambda x: x[1].N, reverse=True):
            kind = action[0]
            obj = action[1]
            move_name = getattr(obj, 'id', getattr(obj, 'species', 'unknown'))
            pct = (child.N / total_visits * 100) if total_visits > 0 else 0
            print(f"  {move_name:20s}: {child.N:3d} visits ({pct:5.1f}%)")
        
    except Exception as e:
        print(f"\nMCTS Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the test
    test_random_randbats_matchup()
    
    print("\n" + "=" * 80)
    print("Run this script again to see a different matchup!")
    print("=" * 80)