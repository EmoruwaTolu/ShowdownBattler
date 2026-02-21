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
    create_mock_battle as _create_mock_battle_orig,
    score_move,
    estimate_damage_fraction,
)
from visualize_tree_depth import visualize_tree_depth

from bot.scoring.switch_score import score_switch as score_switch_core

def score_switch(target, battle, ctx):
    s = score_switch_core(target, battle, ctx)
    # uncomment to log only top-level calls if spammy:
    print("switch", getattr(target, "species", "?"), "score", s)
    return s

# Patch create_mock_battle to add gen attribute
def create_mock_battle(*args, **kwargs):
    """Wrapper that adds gen attribute to battle mock"""
    battle = _create_mock_battle_orig(*args, **kwargs)
    battle.gen = 9  # Add this for eval.py
    return battle

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


def test_6v6_team_awareness():
    """
    Test MCTS behavior in a 6v6 scenario.
    
    Scenarios to test:
    1. Win condition preservation (don't sacrifice setup sweeper)
    2. Numbers advantage recognition (being up 5v2 should matter)
    3. Unseen Pokemon awareness (don't assume it's just 1v1)
    4. Strategic switching (preserve important Pokemon)
    """
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 25 + "6v6 TEAM AWARENESS TEST" + " " * 29 + "║")
    print("╚" + "=" * 78 + "╝")
    
    gen_data = GenData.from_gen(9)
    randbats_data = load_randbats_data()
    
    # Define test scenarios
    scenarios = [
        # {
        #     "name": "Win Condition Preservation",
        #     "description": "Boosted sweeper should be preserved, not sacrificed",
        #     "our_team": ["Dragonite", "Garchomp", "Toxapex", "Corviknight", "Heatran", "Weavile"],
        #     "our_hp": [0.60, 1.00, 1.00, 0.30, 0.80, 0.15],  # Dragonite at 60% (boosted)
        #     "our_boosts": [{"atk": 2, "spe": 2}, {}, {}, {}, {}, {}],  # Dragonite +2/+2
        #     "our_active": 0,  # Dragonite is active
        #     "opp_team": ["Landorus-Therian", "Clefable", "Greninja", "Iron Valiant", "Gholdengo", "Zapdos"],
        #     "opp_hp": [1.00, 0.90, 0.85, 0.00, 0.00, 0.00],  # 3 alive, 3 fainted
        #     "opp_active": 0,  # Landorus active
        #     "expected": "Should preserve +2 Dragonite (our win condition) even at 60% HP"
        # },
        # {
        #     "name": "Numbers Advantage",
        #     "description": "Being up 5v2 should encourage aggressive play",
        #     "our_team": ["Garchomp", "Dragonite", "Toxapex", "Corviknight", "Heatran", "Weavile"],
        #     "our_hp": [0.80, 0.70, 0.90, 0.60, 0.75, 0.85],  # All healthy
        #     "our_boosts": [{}, {}, {}, {}, {}, {}],
        #     "our_active": 0,  # Garchomp active
        #     "opp_team": ["Landorus-Therian", "Clefable", "Greninja", "Iron Valiant", "Gholdengo", "Zapdos"],
        #     "opp_hp": [0.65, 0.40, 0.00, 0.00, 0.00, 0.00],  # Only 2 alive
        #     "opp_active": 0,
        #     "expected": "Should play aggressively - we're up 5v2!"
        # },
        {
            "name": "Unseen Pokemon Caution",
            "description": "Should account for unseen opponent Pokemon",
            "our_team": ["Dragonite", "Garchomp", "Toxapex", "Corviknight", "Heatran", "Weavile"],
            "our_hp": [0.90, 0.85, 0.80, 0.75, 0.70, 0.65],  # All healthy
            "our_boosts": [{}, {}, {}, {}, {}, {}],
            "our_active": 0,
            "opp_team": ["Landorus-Therian", "Clefable"],  # Only 2 seen (4 unseen!)
            "opp_hp": [0.70, 0.50],
            "opp_active": 0,
            "expected": "Should be cautious - opponent has 4 unseen Pokemon"
        },
        # {
        #     "name": "Strategic Switching",
        #     "description": "Should switch to preserve low HP important Pokemon",
        #     "our_team": ["Dragonite", "Garchomp", "Toxapex", "Corviknight", "Heatran", "Weavile"],
        #     "our_hp": [0.25, 0.90, 0.85, 0.80, 0.75, 0.70],  # Dragonite low
        #     "our_boosts": [{"atk": 3, "spe": 3}, {}, {}, {}, {}, {}],  # But heavily boosted!
        #     "our_active": 0,  # Dragonite active at 25% HP
        #     "opp_team": ["Weavile", "Clefable", "Greninja", "Kingambit", "Great Tusk", "Zapdos"],
        #     "opp_hp": [1.00, 0.90, 0.85, 0.80, 0.75, 0.70],  # All healthy
        #     "opp_active": 0,  # Weavile (threatens with Ice Shard)
        #     "expected": "Should switch out +3 Dragonite to preserve win condition"
        # },
    ]
    
    for scenario in scenarios:
        run_6v6_scenario(scenario, gen_data, randbats_data)


def run_6v6_scenario(scenario: dict, gen_data, randbats_data):
    """Run a single 6v6 test scenario"""
    
    # print("\n" + "=" * 80)
    # print(f"SCENARIO: {scenario['name']}")
    # print("=" * 80)
    # print(f"Description: {scenario['description']}")
    # print(f"Expected: {scenario['expected']}")
    
    # Create our team
    our_team_mons = []
    our_team_dict = {}
    
    for i, species in enumerate(scenario['our_team']):
        mon, role, moves, ability, item = create_pokemon_from_randbats(
            species, randbats_data[species], gen_data
        )
        mon._identifier_string = f"p1: {species}"
        mon.current_hp_fraction = scenario['our_hp'][i]
        
        # Set boosts
        boosts = scenario['our_boosts'][i]
        mon.boosts = boosts if boosts else {}
        
        our_team_mons.append(mon)
        our_team_dict[f"p1: {species}"] = mon
    
    # Create opponent team
    opp_team_mons = []
    opp_team_dict = {}
    
    for i, species in enumerate(scenario['opp_team']):
        mon, role, moves, ability, item = create_pokemon_from_randbats(
            species, randbats_data[species], gen_data
        )
        mon._identifier_string = f"p2: {species}"
        mon.current_hp_fraction = scenario['opp_hp'][i]
        
        opp_team_mons.append(mon)
        opp_team_dict[f"p2: {species}"] = mon
    
    # Set active Pokemon
    our_active = our_team_mons[scenario['our_active']]
    opp_active = opp_team_mons[scenario['opp_active']]
    
    # Display teams
    # print("\n" + "-" * 80)
    # print("OUR TEAM")
    # print("-" * 80)
    # for i, mon in enumerate(our_team_mons):
    #     species = scenario['our_team'][i]
    #     hp = scenario['our_hp'][i]
    #     boosts = scenario['our_boosts'][i]
    #     active_marker = "← ACTIVE" if i == scenario['our_active'] else ""
        
    #     boost_str = ""
    #     if boosts:
    #         boost_parts = [f"{stat}{val:+d}" for stat, val in boosts.items() if val != 0]
    #         if boost_parts:
    #             boost_str = f" [{', '.join(boost_parts)}]"
        
    #     print(f"  {i+1}. {species:20s} {hp:>5.0%} HP{boost_str} {active_marker}")
        
    #     # Display moveset
    #     moves_list = list(mon.moves.values())
    #     if moves_list:
    #         move_names = [getattr(m, 'id', 'unknown') for m in moves_list]
    #         print(f"      Moves: {', '.join(move_names)}")
        
    #     # Display ability and item
    #     ability = getattr(mon, 'ability', None)
    #     item = getattr(mon, 'item', None)
    #     if ability or item:
    #         details = []
    #         if ability:
    #             details.append(f"Ability: {ability}")
    #         if item:
    #             details.append(f"Item: {item}")
    #         print(f"      {' | '.join(details)}")
    
    # print("\n" + "-" * 80)
    # print("OPPONENT TEAM")
    # print("-" * 80)
    # for i, mon in enumerate(opp_team_mons):
    #     species = scenario['opp_team'][i]
    #     hp = scenario['opp_hp'][i]
    #     active_marker = "← ACTIVE" if i == scenario['opp_active'] else ""
    #     print(f"  {i+1}. {species:20s} {hp:>5.0%} HP {active_marker}")
        
    #     # Display moveset
    #     moves_list = list(mon.moves.values())
    #     if moves_list:
    #         move_names = [getattr(m, 'id', 'unknown') for m in moves_list]
    #         print(f"      Moves: {', '.join(move_names)}")
        
    #     # Display ability and item
    #     ability = getattr(mon, 'ability', None)
    #     item = getattr(mon, 'item', None)
    #     if ability or item:
    #         details = []
    #         if ability:
    #             details.append(f"Ability: {ability}")
    #         if item:
    #             details.append(f"Item: {item}")
    #         print(f"      {' | '.join(details)}")
    
    # unseen = 6 - len(scenario['opp_team'])
    # if unseen > 0:
    #     print(f"  ... and {unseen} unseen Pokemon")
    
    # Team value analysis
    print("\n" + "-" * 80)
    print("TEAM VALUE ANALYSIS")
    print("-" * 80)
    
    # Import the actual team_value function from eval.py
    from bot.mcts.eval import team_value
    
    # Calculate our team value using ACTUAL eval.py function
    gen = 9
    our_hp_map = {id(mon): hp for mon, hp in zip(our_team_mons, scenario['our_hp'])}
    our_boosts_map = {id(mon): boosts for mon, boosts in zip(our_team_mons, scenario['our_boosts'])}
    
    our_value = team_value(our_team_mons, our_hp_map, our_boosts_map, gen)
    
    opp_hp_map = {id(mon): hp for mon, hp in zip(opp_team_mons, scenario['opp_hp'])}
    opp_boosts_map = {id(mon): {} for mon in opp_team_mons}  # Opponent boosts unknown
    
    opp_value_known = team_value(opp_team_mons, opp_hp_map, opp_boosts_map, gen)
    opp_known_count = len(scenario['opp_team'])
    
    # Use dynamic unseen discount from eval.py
    from bot.mcts.eval import opp_unseen_value
    opp_value_unseen = opp_unseen_value(opp_known_count, 6)
    opp_value_total = opp_value_known + opp_value_unseen
    
    # print(f"\nOur team value: {our_value:.2f}")
    # print(f"  Breakdown (using opponent belief system):")
    # for i, mon in enumerate(our_team_mons):
    #     species = scenario['our_team'][i]
    #     hp = scenario['our_hp'][i]
    #     boosts = scenario['our_boosts'][i]
        
    #     if hp <= 0:
    #         continue
        
    #     # Get ACTUAL role weight from belief system
    #     role_w = expected_role_weight_for_mon(mon, gen)
        
    #     # Boost multiplier
    #     max_boost = max(boosts.values()) if boosts else 0
    #     if hp < 0.35:
    #         max_boost = min(max_boost, 2)
        
    #     if max_boost >= 4:
    #         boost_mult = 1.18
    #     elif max_boost >= 2:
    #         boost_mult = 1.12
    #     elif max_boost >= 1:
    #         boost_mult = 1.06
    #     else:
    #         boost_mult = 1.00
        
    #     # Low HP multiplier
    #     if hp < 0.20:
    #         hp_mult = 0.35 if role_w > 1.06 else 0.45
    #     elif hp < 0.35:
    #         hp_mult = 0.55 if role_w > 1.06 else 0.65
    #     elif hp < 0.55:
    #         hp_mult = 0.80
    #     else:
    #         hp_mult = 1.00
        
    #     value = hp * role_w * boost_mult * hp_mult
        
    #     details = f"role:{role_w:.2f}"
    #     if boost_mult != 1.0:
    #         details += f" × boost:{boost_mult:.2f}"
    #     if hp_mult != 1.0:
    #         details += f" × hp_mult:{hp_mult:.2f}"
        
    #     print(f"    {species:20s}: {hp:.2f} → {value:.2f} ({details})")
    
    print(f"\nOpponent team value: {opp_value_total:.2f}")
    print(f"  Known ({opp_known_count} Pokemon): {opp_value_known:.2f}")
    
    opp_unseen_count = 6 - opp_known_count
    if opp_unseen_count > 0:
        # Show the discount calculation
        if opp_known_count <= 1:
            discount = 0.95
        elif opp_known_count <= 3:
            discount = 0.90
        elif opp_known_count <= 5:
            discount = 0.80
        else:
            discount = 0.0
        
        print(f"  Unseen ({opp_unseen_count} Pokemon): {opp_value_unseen:.2f}")
        print(f"    (Using dynamic discount: {discount:.2f} based on {opp_known_count} known)")
    
    advantage = our_value - opp_value_total
    print(f"\nTeam value advantage: {advantage:+.2f}")
    if advantage > 1.0:
        print("  → We're significantly ahead")
    elif advantage > 0.3:
        print("  → We're slightly ahead")
    elif advantage > -0.3:
        print("  → Roughly even")
    elif advantage > -1.0:
        print("  → We're slightly behind")
    else:
        print("  → We're significantly behind")
    
    # Alive count and progress
    our_alive = sum(1 for hp in our_hp_map.values() if hp > 0)
    opp_alive_known = sum(1 for hp in opp_hp_map.values() if hp > 0)
    opp_alive_total = opp_alive_known + opp_unseen_count
    ahead = our_alive - opp_alive_total
    
    print(f"\nAlive Pokemon:")
    print(f"  Us: {our_alive}")
    print(f"  Opponent: {opp_alive_total} ({opp_alive_known} known + {opp_unseen_count} unseen)")
    print(f"  Ahead by: {ahead:+d}")
    
    if ahead >= 2:
        print(f"  → We're ahead! (Dynamic eval will favor aggression)")
    elif ahead <= -2:
        print(f"  → We're behind! (Dynamic eval will favor safety)")
    
    # Progress metric
    opp_kos = 6 - opp_alive_total
    progress = opp_kos / 6.0
    print(f"\nProgress toward winning:")
    print(f"  Opponent KOs: {opp_kos}/6 ({progress:.1%})")
    if ahead >= 2:
        print(f"  → Progress term will contribute {0.15 * progress:.3f} to evaluation")
    
    # Healthy Pokemon count
    our_healthy = sum(1 for hp in scenario['our_hp'] if hp > 0.55)
    opp_healthy = sum(1 for hp in scenario['opp_hp'] if hp > 0.55) + (6 - len(scenario['opp_team']))
    
    print(f"\nHealthy Pokemon (>55% HP):")
    print(f"  Us: {our_healthy}")
    print(f"  Opponent: {opp_healthy} ({sum(1 for hp in scenario['opp_hp'] if hp > 0.55)} known + {6 - len(scenario['opp_team'])} unseen)")
    print(f"  Numbers advantage: {our_healthy - opp_healthy:+d}")
    
    # Create battle
    battle = create_mock_battle(
        active_identifier=our_active._identifier_string,
        active=our_active,
        opponent_identifier=opp_active._identifier_string,
        opponent=opp_active,
        team=our_team_dict,
        opponent_team=opp_team_dict
    )
    
    # Mark opponent team size
    battle.opponent_team_size = 6
    
    ctx_me = EvalContext(me=our_active, opp=opp_active, battle=battle, cache={})
    ctx_opp = EvalContext(me=opp_active, opp=our_active, battle=battle, cache={})
    
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
    
    battle.available_moves = list(our_active.moves.values())
    battle.available_switches = [mon for mon in our_team_mons if mon != our_active and mon.current_hp_fraction > 0]
    
    from bot.mcts.shadow_state import ShadowState
    
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
        
        # print("\nMCTS Results:")
        # print(f"  Simulations: {stats['root'].N}")
        # print(f"  Picked: {picked[0]} {getattr(picked[1], 'id', getattr(picked[1], 'species', 'unknown'))}")
        
        # # Action distribution
        # print("\n  Action Distribution:")
        
        moves = []
        switches = []
        
        for action, child in sorted(stats['root'].children.items(), 
                                   key=lambda x: x[1].N, reverse=True):
            kind, obj = action
            name = getattr(obj, 'id', getattr(obj, 'species', 'unknown'))
            visits = child.N
            q_val = child.Q
            pct = visits / stats['root'].N * 100
            
            if kind == 'move':
                moves.append((name, visits, q_val, pct))
            else:
                switches.append((name, visits, q_val, pct))
        
        # if moves:
        #     print("Moves:")
        #     for name, visits, q_val, pct in moves[:5]:
        #         print(f"    {name:20s}: {visits:4d} visits ({pct:5.1f}%), Q={q_val:+.3f}")
        
        if switches:
            print("Switches:")
            for name, visits, q_val, pct in switches[:5]:
                print(f"    → {name:20s}: {visits:4d} visits ({pct:5.1f}%), Q={q_val:+.3f}")
        
        # Interpretation
        # print("\n" + "-" * 80)
        # print("INTERPRETATION")
        # print("-" * 80)
        
        # top_action = max(stats['root'].children.items(), key=lambda x: x[1].N)
        # top_kind, top_obj = top_action[0]
        # top_name = getattr(top_obj, 'id', getattr(top_obj, 'species', 'unknown'))
        # top_visits = top_action[1].N
        
        # if top_kind == 'switch':
        #     print(f"\n✓ MCTS chose to SWITCH to {top_name}")
        #     print(f"  This suggests:")
        #     print(f"  - Current active Pokemon should be preserved")
        #     print(f"  - {top_name} is a better matchup")
        #     print(f"  - Team-aware decision making")
        # else:
        #     print(f"\n✓ MCTS chose to use {top_name}")
        #     print(f"  This suggests:")
        #     print(f"  - Current matchup is favorable")
        #     print(f"  - Pressing advantage is correct")
        
        # # Check if it matches expectation
        # print(f"\nExpected behavior: {scenario['expected']}")
        
        # # Tree visualization
        # print("\n" + "-" * 80)
        # print("TREE VISUALIZATION (Top Actions)")
        # print("-" * 80)
        # visualize_tree_depth(stats['root'], max_depth=2)
        
    except Exception as e:
        print(f"\n❌ MCTS Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_6v6_team_awareness()