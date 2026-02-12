"""
Minimal test to reproduce the forced switch bug.
"""

import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poke_env.data import GenData
from bot.mcts.shadow_state import ShadowState
from bot.model.ctx import EvalContext
from bot.scoring.move_score import score_move
from bot.scoring.switch_score import score_switch
from bot.scoring.damage_score import estimate_damage_fraction

from test_random_randbats import load_randbats_data, create_pokemon_from_randbats
from test_real_heuristics import create_mock_battle

def test_forced_switch_bug():
    """
    Reproduce the bug where using a non-pivot move causes forced switch.
    """
    
    gen_data = GenData.from_gen(9)
    randbats_data = load_randbats_data()
    
    # Create Garchomp
    garchomp, _, _, _, _ = create_pokemon_from_randbats(
        "Garchomp", randbats_data["Garchomp"], gen_data
    )
    garchomp._identifier_string = "p1: Garchomp"
    garchomp.current_hp_fraction = 0.58  # 58% HP
    
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
    
    # Apply +2 attack to Garchomp
    state = state._apply_boost_changes({'atk': 2}, 'me', garchomp)
    
    print("="*80)
    print("FORCED SWITCH BUG TEST")
    print("="*80)
    
    # Find Earthquake move
    earthquake = None
    for move in garchomp.moves.values():
        if move.id == "earthquake":
            earthquake = move
            break
    
    if not earthquake:
        print("ERROR: Garchomp doesn't have Earthquake!")
        return
    
    print(f"\nInitial state:")
    print(f"  Our active: {state.my_active.species} at {state.my_active_hp():.0%} HP")
    print(f"  Opp active: {state.opp_active.species} at {state.opp_active_hp():.0%} HP")
    print(f"  Our boosts: {state.my_boosts.get(id(state.my_active), {})}")
    
    print(f"\nAction: We use Earthquake")
    print(f"  Earthquake is pivot move: {hasattr(earthquake, 'pivot') or earthquake.id in ['uturn', 'voltswitch', 'flipturn', 'partingshot', 'batonpass']}")
    
    # Step with Earthquake
    rng = random.Random(42)
    my_action = ("move", earthquake)
    
    # Manually trace through what step() does
    print(f"\n--- TRACING step() execution ---")
    
    print(f"1. choose_opp_action()")
    opp_action = state.choose_opp_action(rng)
    print(f"   Opponent chose: {opp_action[0]} {getattr(opp_action[1], 'species', getattr(opp_action[1], 'id', 'unknown'))}")
    
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
            print(f"    Opponent action: {opp_action[0]} {getattr(opp_action[1], 'species', getattr(opp_action[1], 'id', 'unknown'))}")
            print(f"    State before: my_active={s.my_active.species} opp_active={s.opp_active.species}")
            print(f"    HP before:    my={s.my_active_hp():.0%} opp={s.opp_active_hp():.0%}")
            s = s._apply_opp_action(opp_action, rng)
            print(f"    State after:  my_active={s.my_active.species} opp_active={s.opp_active.species}")
            print(f"    HP after:     my={s.my_active_hp():.0%} opp={s.opp_active_hp():.0%}")
        else:
            print(f"\n3b. SKIPPED - opponent's active is at 0% HP")
    else:
        print(f"\n3a. Opponent goes first - _apply_opp_action()")
        print(f"    Opponent action: {opp_action[0]} {getattr(opp_action[1], 'species', getattr(opp_action[1], 'id', 'unknown'))}")
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
    
    print(f"\n4. _apply_end_of_turn_chip()")
    if not s.is_terminal():
        print(f"   State before: my_active={s.my_active.species} opp_active={s.opp_active.species}")
        print(f"   HP before:    my={s.my_active_hp():.0%} opp={s.opp_active_hp():.0%}")
        s = s._apply_end_of_turn_chip()
        print(f"   State after:  my_active={s.my_active.species} opp_active={s.opp_active.species}")
        print(f"   HP after:     my={s.my_active_hp():.0%} opp={s.opp_active_hp():.0%}")
    
    next_state = s
    
    print(f"\n--- END OF TRACE ---\n")
    
    print(f"\nAfter step:")
    print(f"  Our active: {next_state.my_active.species} at {next_state.my_active_hp():.0%} HP")
    print(f"  Opp active: {next_state.opp_active.species} at {next_state.opp_active_hp():.0%} HP")
    
    print(f"\nBUG CHECK:")
    if state.my_active != next_state.my_active:
        print(f"  ✗ BUG DETECTED! Our Pokemon changed from {state.my_active.species} to {next_state.my_active.species}")
        print(f"    Previous HP: {state.my_active_hp():.0%}")
        print(f"    Garchomp did NOT faint, but was forced to switch!")
    else:
        print(f"  ✓ No bug - Pokemon stayed the same")
    
    if state.opp_active != next_state.opp_active:
        print(f"  Opponent switched from {state.opp_active.species} to {next_state.opp_active.species}")
        print(f"    Previous HP: {state.opp_active_hp():.0%}")
        
        if state.opp_active_hp() > 0.01:
            print(f"    (Voluntary switch - opponent wasn't KO'd)")
        else:
            print(f"    (Forced switch - opponent was KO'd)")

if __name__ == "__main__":
    test_forced_switch_bug()