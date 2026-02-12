"""
Test the exact scenario from MCTS output where Stone Edge causes forced switch.
"""

import sys
import os
import random

if 'bot.mcts.shadow_state' in sys.modules:
    del sys.modules['bot.mcts.shadow_state']

# Now import fresh
from bot.mcts.shadow_state import ShadowState
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poke_env.data import GenData
from bot.model.ctx import EvalContext
from bot.scoring.move_score import score_move
from bot.scoring.switch_score import score_switch
from bot.scoring.damage_score import estimate_damage_fraction

from test_random_randbats import load_randbats_data, create_pokemon_from_randbats
from test_real_heuristics import create_mock_battle

def test_stoneedge_scenario():
    """
    Reproduce the exact scenario:
    D1: Garchomp 34% [at+2] vs Clefable 40%
      -> We use Stone Edge
      -> Result: Weavile 85% vs Landorus 8%
    """
    
    gen_data = GenData.from_gen(9)
    randbats_data = load_randbats_data()
    
    # Create Garchomp with +2 attack
    garchomp, _, _, _, _ = create_pokemon_from_randbats(
        "Garchomp", randbats_data["Garchomp"], gen_data
    )
    garchomp._identifier_string = "p1: Garchomp"
    garchomp.current_hp_fraction = 0.34
    
    # Create Weavile  
    weavile, _, _, _, _ = create_pokemon_from_randbats(
        "Weavile", randbats_data["Weavile"], gen_data
    )
    weavile._identifier_string = "p1: Weavile"
    weavile.current_hp_fraction = 0.85
    
    # Create Clefable
    clefable, _, _, _, _ = create_pokemon_from_randbats(
        "Clefable", randbats_data["Clefable"], gen_data
    )
    clefable._identifier_string = "p2: Clefable"
    clefable.current_hp_fraction = 0.40
    
    # Create Landorus
    landorus, _, _, _, _ = create_pokemon_from_randbats(
        "Landorus-Therian", randbats_data["Landorus-Therian"], gen_data
    )
    landorus._identifier_string = "p2: Landorus-Therian"
    landorus.current_hp_fraction = 0.65
    
    # Setup teams
    our_team_dict = {
        "p1: Garchomp": garchomp,
        "p1: Weavile": weavile,
    }
    
    opp_team_dict = {
        "p2: Clefable": clefable,
        "p2: Landorus-Therian": landorus,
    }
    
    # Create battle
    battle = create_mock_battle(
        active_identifier=garchomp._identifier_string,
        active=garchomp,
        opponent_identifier=clefable._identifier_string,
        opponent=clefable,
        team=our_team_dict,
        opponent_team=opp_team_dict
    )
    battle.gen = 9
    
    ctx_me = EvalContext(me=garchomp, opp=clefable, battle=battle, cache={})
    ctx_opp = EvalContext(me=clefable, opp=garchomp, battle=battle, cache={})
    
    # Create shadow state
    state = ShadowState.from_battle(
        battle=battle,
        ctx_me=ctx_me,
        ctx_opp=ctx_opp,
        score_move_fn=score_move,
        score_switch_fn=score_switch,
        dmg_fn=estimate_damage_fraction,
    )

    print(f"\n=== HP DICTIONARY DEBUG ===")
    print(f"Garchomp ID: {id(garchomp)}")
    print(f"Garchomp current_hp_fraction: {garchomp.current_hp_fraction}")
    from bot.scoring.helpers import hp_frac
    print(f"hp_frac(garchomp): {hp_frac(garchomp)}")
    print(f"state.my_hp[id(garchomp)]: {state.my_hp.get(id(garchomp), 'NOT FOUND')}")
    print(f"state.my_active is garchomp: {state.my_active is garchomp}")
    print(f"state.my_active ID: {id(state.my_active)}")
    print("===========================\n")
    
    # Apply +2 attack to Garchomp
    state = state._apply_boost_changes({'atk': 2}, 'me', garchomp)

    print(f"\n=== AFTER BOOST DEBUG ===")
    print(f"Original Garchomp ID: {id(garchomp)}")
    print(f"state.my_active ID: {id(state.my_active)}")
    print(f"IDs match: {id(garchomp) == id(state.my_active)}")
    print(f"state.my_hp[id(garchomp)]: {state.my_hp.get(id(garchomp), 'NOT FOUND')}")
    print(f"state.my_hp[id(state.my_active)]: {state.my_hp.get(id(state.my_active), 'NOT FOUND')}")
    print(f"All my_hp keys: {list(state.my_hp.keys())}")
    print("=========================\n")
    
    print("="*80)
    print("STONE EDGE SCENARIO TEST")
    print("="*80)
    
    # Find Stone Edge move
    stone_edge = None
    for move in garchomp.moves.values():
        if move.id == "stoneedge":
            stone_edge = move
            break
    
    if not stone_edge:
        print("ERROR: Garchomp doesn't have Stone Edge!")
        # Try other moves
        print(f"Available moves: {[m.id for m in garchomp.moves.values()]}")
        return
    
    print(f"\nInitial state:")
    print(f"  Our active: {state.my_active.species} at {state.my_active_hp():.0%} HP [at+2]")
    print(f"  Opp active: {state.opp_active.species} at {state.opp_active_hp():.0%} HP")
    
    print(f"\nAction: We use Stone Edge")
    
    # Step with Stone Edge
    rng = random.Random(42)
    my_action = ("move", stone_edge)
    
    # Detailed trace
    print(f"\n--- TRACING step() execution ---")
    
    print(f"1. choose_opp_action()")
    opp_action = state.choose_opp_action(rng)
    opp_name = getattr(opp_action[1], 'species', getattr(opp_action[1], 'id', 'unknown'))
    print(f"   Opponent chose: {opp_action[0]} {opp_name}")
    
    print(f"\n2. _order_for_turn()")
    order = state._order_for_turn(my_action, opp_action, rng)
    print(f"   Order: {'+1 (we go first)' if order == 1 else '-1 (opponent goes first)'}")
    
    s = state
    if order == +1:
        print(f"\n3a. We go first - _apply_my_action()")
        print(f"    State before: my_active={s.my_active.species} opp_active={s.opp_active.species}")
        print(f"    HP before:    my={s.my_active_hp():.0%} opp={s.opp_active_hp():.0%}")
        s = s._apply_my_action(my_action, rng)
        print(f"    State after:  my_active={s.my_active.species} opp_active={s.opp_active.species}")
        print(f"    HP after:     my={s.my_active_hp():.0%} opp={s.opp_active_hp():.0%}")
        
        if s.opp_active_hp() > 0.0:
            print(f"\n3b. Opponent goes second - _apply_opp_action()")
            print(f"    Opponent action: {opp_action[0]} {opp_name}")
            print(f"    State before: my_active={s.my_active.species} opp_active={s.opp_active.species}")
            print(f"    HP before:    my={s.my_active_hp():.0%} opp={s.opp_active_hp():.0%}")
            s = s._apply_opp_action(opp_action, rng)
            print(f"    State after:  my_active={s.my_active.species} opp_active={s.opp_active.species}")
            print(f"    HP after:     my={s.my_active_hp():.0%} opp={s.opp_active_hp():.0%}")
        else:
            print(f"\n3b. SKIPPED - opponent's active is at 0% HP")
    else:
        print(f"\n3a. Opponent goes first - _apply_opp_action()")
        print(f"    Opponent action: {opp_action[0]} {opp_name}")
        print(f"    State before: my_active={s.my_active.species} opp_active={s.opp_active.species}")
        print(f"    HP before:    my={s.my_active_hp():.0%} opp={s.opp_active_hp():.0%}")
        s = s._apply_opp_action(opp_action, rng)
        print(f"    State after:  my_active={s.my_active.species} opp_active={s.opp_active.species}")
        print(f"    HP after:     my={s.my_active_hp():.0%} opp={s.opp_active_hp():.0%}")
        
        if s.my_active_hp() > 0.0:
            print(f"\n3b. We go second - _apply_my_action()")
            print(f"    State before: my_active={s.my_active.species} opp_active={s.opp_active.species}")
            print(f"    HP before:    my={s.my_active_hp():.0%} opp={s.opp_active_hp():.0%}")
            s = s._apply_my_action(my_action, rng)
            print(f"    State after:  my_active={s.my_active.species} opp_active={s.opp_active.species}")
            print(f"    HP after:     my={s.my_active_hp():.0%} opp={s.opp_active_hp():.0%}")
        else:
            print(f"\n3b. SKIPPED - our active is at 0% HP")
    
    next_state = s
    
    print(f"\n--- END OF TRACE ---\n")
    
    print(f"Final state:")
    print(f"  Our active: {next_state.my_active.species} at {next_state.my_active_hp():.0%} HP")
    print(f"  Opp active: {next_state.opp_active.species} at {next_state.opp_active_hp():.0%} HP")
    
    print(f"\nEXPLANATION:")
    if state.my_active != next_state.my_active:
        print(f"  Our Pokemon changed from {state.my_active.species} to {next_state.my_active.species}")
        if state.my_active_hp() > 0.01:
            print(f"  ✗ BUG: {state.my_active.species} was at {state.my_active_hp():.0%} HP but switched anyway!")
        else:
            print(f"  ✓ Correct: {state.my_active.species} fainted")
    
    if state.opp_active != next_state.opp_active:
        print(f"  Opponent changed from {state.opp_active.species} to {next_state.opp_active.species}")
        if state.opp_active_hp() > 0.01:
            print(f"  Opponent voluntarily switched")
        else:
            print(f"  Opponent's {state.opp_active.species} fainted")

    print(f"Auto-switch check: terminal={next_state.is_terminal()}, best_switch={next_state._best_opp_switch()}")
    print(f"\n--- Now calling actual step() function ---")
    next_state = state.step(my_action, rng=rng)
    print(f"--- step() complete ---\n")

    print(f"Final state (from step()):")
    print(f"  Our active: {next_state.my_active.species} at {next_state.my_active_hp():.0%} HP")
    print(f"  Opp active: {next_state.opp_active.species} at {next_state.opp_active_hp():.0%} HP")

if __name__ == "__main__":
    test_stoneedge_scenario()