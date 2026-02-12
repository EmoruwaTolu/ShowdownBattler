"""
Diagnostic: Check if score_setup_move actually sees the boosts
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poke_env.data import GenData
from bot.mcts.shadow_state import ShadowState
from bot.scoring.move_score import score_setup_move
from test_random_randbats import load_randbats_data, create_pokemon_from_randbats
from test_real_heuristics import create_mock_battle

def test_boost_visibility():
    """Test if score_setup_move can see current boosts"""
    
    gen_data = GenData.from_gen(9)
    randbats_data = load_randbats_data()
    
    # Create Garchomp
    garchomp, _, _, _, _ = create_pokemon_from_randbats(
        "Garchomp", randbats_data["Garchomp"], gen_data
    )
    garchomp._identifier_string = "p1: Garchomp"
    garchomp.current_hp_fraction = 0.80
    
    # Create opponent
    landorus, _, _, _, _ = create_pokemon_from_randbats(
        "Landorus-Therian", randbats_data["Landorus-Therian"], gen_data
    )
    landorus._identifier_string = "p2: Landorus-Therian"
    landorus.current_hp_fraction = 0.65
    
    # Create battle
    our_team_dict = {"p1: Garchomp": garchomp}
    opp_team_dict = {"p2: Landorus-Therian": landorus}
    
    battle = create_mock_battle(
        active_identifier=garchomp._identifier_string,
        active=garchomp,
        opponent_identifier=landorus._identifier_string,
        opponent=landorus,
        team=our_team_dict,
        opponent_team=opp_team_dict
    )
    battle.gen = 9
    
    # Find Swords Dance move
    swords_dance = None
    for move in garchomp.moves.values():
        if move.id == "swordsdance":
            swords_dance = move
            break
    
    if not swords_dance:
        print("ERROR: Garchomp doesn't have Swords Dance!")
        return
    
    from bot.model.ctx import EvalContext
    
    print("=" * 80)
    print("BOOST VISIBILITY TEST")
    print("=" * 80)
    
    # Test 1: Score at +0 boosts
    print("\n--- TEST 1: Garchomp at +0 attack ---")
    garchomp.boosts = {}  # No boosts
    ctx = EvalContext(me=garchomp, opp=landorus, battle=battle, cache={})
    
    score_0 = score_setup_move(swords_dance, battle, ctx)
    print(f"Pokemon.boosts: {garchomp.boosts}")
    print(f"Swords Dance score: {score_0:.2f}")
    
    # Test 2: Score at +2 attack
    print("\n--- TEST 2: Garchomp at +2 attack ---")
    garchomp.boosts = {'atk': 2, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0}
    ctx = EvalContext(me=garchomp, opp=landorus, battle=battle, cache={})
    
    score_2 = score_setup_move(swords_dance, battle, ctx)
    print(f"Pokemon.boosts: {garchomp.boosts}")
    print(f"Swords Dance score: {score_2:.2f}")
    
    # Test 3: Score at +4 attack
    print("\n--- TEST 3: Garchomp at +4 attack ---")
    garchomp.boosts = {'atk': 4, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0}
    ctx = EvalContext(me=garchomp, opp=landorus, battle=battle, cache={})
    
    score_4 = score_setup_move(swords_dance, battle, ctx)
    print(f"Pokemon.boosts: {garchomp.boosts}")
    print(f"Swords Dance score: {score_4:.2f}")
    
    print("\n" + "=" * 80)
    print("EXPECTED RESULTS:")
    print("=" * 80)
    print(f"Score at +0: ~60 (base value)")
    print(f"Score at +2: ~36 (stages +3,+4 with 0.7×, 0.5× multipliers)")
    print(f"Score at +4: ~12 (stages +5,+6 with 0.3×, 0.1× multipliers)")
    print()
    print("ACTUAL RESULTS:")
    print(f"Score at +0: {score_0:.2f}")
    print(f"Score at +2: {score_2:.2f}  {'✓ CORRECT' if score_2 < score_0 * 0.7 else '✗ BUG - should be lower!'}")
    print(f"Score at +4: {score_4:.2f}  {'✓ CORRECT' if score_4 < score_2 * 0.5 else '✗ BUG - should be lower!'}")
    
    # Now test with ShadowState patching
    print("\n" + "=" * 80)
    print("SHADOW STATE PATCHING TEST")
    print("=" * 80)
    
    from bot.scoring.move_score import score_move
    from bot.scoring.switch_score import score_switch
    from bot.scoring.damage_score import estimate_damage_fraction
    
    # Create shadow state with +2 attack
    state = ShadowState.from_battle(
        battle=battle,
        ctx_me=ctx,
        ctx_opp=EvalContext(me=landorus, opp=garchomp, battle=battle, cache={}),
        score_move_fn=score_move,
        score_switch_fn=score_switch,
        dmg_fn=estimate_damage_fraction,
    )
    
    # Manually set boosts
    state = state._apply_boost_changes({'atk': 2}, 'me', garchomp)
    
    print(f"\nShadowState.my_boosts[id(garchomp)]: {state.my_boosts.get(id(garchomp), {})}")
    print(f"garchomp.boosts (before patch): {garchomp.boosts}")
    
    # Score WITH patching
    print("\n--- Scoring WITH _patched_boosts() ---")
    with state._patched_boosts():
        print(f"garchomp.boosts (inside patch): {garchomp.boosts}")
        ctx_patched = EvalContext(me=garchomp, opp=landorus, battle=battle, cache={})
        score_patched = score_setup_move(swords_dance, battle, ctx_patched)
        print(f"Swords Dance score: {score_patched:.2f}")
    
    print(f"garchomp.boosts (after patch): {garchomp.boosts}")
    
    # Score WITHOUT patching
    print("\n--- Scoring WITHOUT _patched_boosts() ---")
    garchomp.boosts = {}  # Reset
    ctx_unpatched = EvalContext(me=garchomp, opp=landorus, battle=battle, cache={})
    score_unpatched = score_setup_move(swords_dance, battle, ctx_unpatched)
    print(f"garchomp.boosts: {garchomp.boosts}")
    print(f"Swords Dance score: {score_unpatched:.2f}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS:")
    print("=" * 80)
    if score_patched < score_unpatched * 0.7:
        print("✓ _patched_boosts() IS WORKING - score dropped correctly")
    else:
        print("✗ _patched_boosts() NOT WORKING - score didn't drop!")
        print("   This means the patching isn't reaching score_setup_move()")

if __name__ == "__main__":
    test_boost_visibility()