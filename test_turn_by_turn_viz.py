import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poke_env.battle import MoveCategory, PokemonType
from poke_env.data import GenData
from bot.model.ctx import EvalContext
from bot.mcts.search import search, MCTSConfig

from test_real_heuristics import (
    create_mock_pokemon,
    create_mock_move,
    create_mock_battle as _create_mock_battle_orig,
    score_move,
    score_switch,
    estimate_damage_fraction,
)

from test_random_randbats import (
    load_randbats_data,
    create_pokemon_from_randbats,
)

from visualize_turn_by_turn import visualize_turn_by_turn


# Patch create_mock_battle to add gen attribute
def create_mock_battle(*args, **kwargs):
    """Wrapper that adds gen attribute to battle mock"""
    battle = _create_mock_battle_orig(*args, **kwargs)
    battle.gen = 9  # Add this for eval.py
    return battle


def test_switch_in_damage():
    """
    Test case demonstrating switch-in damage clearly.

    Scenario:
    - Garchomp active vs Landorus
    - Garchomp switches to Weavile
    - Landorus attacks on the switch
    - Weavile takes damage
    """
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 22 + "SWITCH-IN DAMAGE SCENARIO" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")

    gen_data = GenData.from_gen(9)
    randbats_data = load_randbats_data()

    # Create our team
    garchomp, _, _, _, _ = create_pokemon_from_randbats(
        "Garchomp", randbats_data["Garchomp"], gen_data
    )
    garchomp._identifier_string = "p1: Garchomp"
    garchomp.current_hp_fraction = 0.80

    weavile, _, _, _, _ = create_pokemon_from_randbats(
        "Weavile", randbats_data["Weavile"], gen_data
    )
    weavile._identifier_string = "p1: Weavile"
    weavile.current_hp_fraction = 0.85  # Healthy before switch

    clefable, _, _, _, _ = create_pokemon_from_randbats(
        "Clefable", randbats_data["Clefable"], gen_data
    )
    clefable._identifier_string = "p2: Clefable"
    clefable.current_hp_fraction = 0.40

    # Create opponent team
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
        "p2: Landorus-Therian": landorus,
        "p2: Clefable": clefable,
    }

    our_team_mons = [garchomp, weavile]
    opp_team_mons = [landorus, clefable]

    # Display initial state
    print("\n" + "-" * 80)
    print("INITIAL STATE")
    print("-" * 80)
    print(f"Active: {garchomp.species} {garchomp.current_hp_fraction:.0%} HP")
    print(f"  vs")
    print(f"Active: {landorus.species} {landorus.current_hp_fraction:.0%} HP")
    print()
    print("Our available switch:")
    print(f"  - {weavile.species} {weavile.current_hp_fraction:.0%} HP")
    print()
    print("Expected behavior:")
    print("  - If we switch to Weavile, Landorus will attack on the switch")
    print("  - Weavile will take damage before acting")
    print("  - Then Weavile can act (e.g., Ice Shard might KO Landorus)")
    print("  - If Landorus faints, opponent forced to switch (e.g., to Clefable)")

    # Create battle
    battle = create_mock_battle(
        active_identifier=garchomp._identifier_string,
        active=garchomp,
        opponent_identifier=landorus._identifier_string,
        opponent=landorus,
        team=our_team_dict,
        opponent_team=opp_team_dict
    )
    battle.opponent_team_size = 6

    ctx_me = EvalContext(me=garchomp, opp=landorus, battle=battle, cache={})
    ctx_opp = EvalContext(me=landorus, opp=garchomp, battle=battle, cache={})

    # Run MCTS
    print("\n" + "-" * 80)
    print("RUNNING MCTS")
    print("-" * 80)

    cfg = MCTSConfig(
        num_simulations=200,
        max_depth=3,
        c_puct=1.6,
        seed=42,
        use_hybrid_expansion=False,
    )

    print("\n=== Testing Switch Score ===")
    weavile_switch_score = score_switch(weavile, battle, ctx_me)
    print(f"Weavile switch score: {weavile_switch_score}")

    battle.available_moves = list(garchomp.moves.values())
    battle.available_switches = [mon for mon in our_team_mons if mon != garchomp and mon.current_hp_fraction > 0]

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

        print(f"\nMCTS picked: {picked[0]} {getattr(picked[1], 'id', getattr(picked[1], 'species', 'unknown'))}")
        print(f"Root visits: {stats['root'].N}")

        # Use the enhanced visualization
        visualize_turn_by_turn(
            stats['root'],
            max_depth=2,
            score_move_fn=score_move,
            score_switch_fn=score_switch,
            battle=battle,
            ctx_me=ctx_me,
            ctx_opp=ctx_opp
        )

        print("\n" + "-" * 80)
        print("INTERPRETATION")
        print("-" * 80)
        print("The enhanced visualization clearly shows:")
        print("1. Heuristic scores (H) - Raw evaluation before MCTS search")
        print("2. MCTS statistics (N, Q, Prior) - What MCTS learned through search")
        print("3. Turn order - When we switch, opponent attacks FIRST (on the switch)")
        print("4. Damage tracking - Our switch-in takes damage BEFORE acting")
        print("5. Forced switches - If opponent faints, they AUTO-SWITCH")
        print("6. HP tracking - New Pokemon come in at their current HP")
        print()
        print("This explains why you see:")
        print("  - Weavile at low HP after switching (took damage on switch)")
        print("  - Clefable at 40% HP (not 100%) because it's a forced switch")

    except Exception as e:
        print(f"\n❌ MCTS Failed: {e}")
        import traceback
        traceback.print_exc()


def test_simple_ko_scenario():
    """
    Simple scenario: one Pokemon KOs another, forcing a switch.
    """
    print("\n\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 25 + "SIMPLE KO SCENARIO" + " " * 33 + "║")
    print("╚" + "=" * 78 + "╝")

    gen_data = GenData.from_gen(9)
    randbats_data = load_randbats_data()

    # Garchomp vs low HP Moltres
    garchomp, _, _, _, _ = create_pokemon_from_randbats(
        "Garchomp", randbats_data["Garchomp"], gen_data
    )
    garchomp._identifier_string = "p1: Garchomp"
    garchomp.current_hp_fraction = 0.90

    moltres, _, _, _, _ = create_pokemon_from_randbats(
        "Moltres", randbats_data["Moltres"], gen_data
    )
    moltres._identifier_string = "p2: Moltres"
    moltres.current_hp_fraction = 0.20  # Low HP, likely to be KO'd

    zapdos, _, _, _, _ = create_pokemon_from_randbats(
        "Zapdos", randbats_data["Zapdos"], gen_data
    )
    zapdos._identifier_string = "p2: Zapdos"
    zapdos.current_hp_fraction = 0.80

    our_team_dict = {"p1: Garchomp": garchomp}
    opp_team_dict = {
        "p2: Moltres": moltres,
        "p2: Zapdos": zapdos,
    }

    our_team_mons = [garchomp]
    opp_team_mons = [moltres, zapdos]

    print("\n" + "-" * 80)
    print("SCENARIO")
    print("-" * 80)
    print(f"Garchomp 90% HP vs Moltres 20% HP")
    print(f"  - Stone Edge likely KOs Moltres")
    print(f"  - Opponent forced to switch to Zapdos")

    battle = create_mock_battle(
        active_identifier=garchomp._identifier_string,
        active=garchomp,
        opponent_identifier=moltres._identifier_string,
        opponent=moltres,
        team=our_team_dict,
        opponent_team=opp_team_dict
    )
    battle.opponent_team_size = 6

    ctx_me = EvalContext(me=garchomp, opp=moltres, battle=battle, cache={})
    ctx_opp = EvalContext(me=moltres, opp=garchomp, battle=battle, cache={})

    cfg = MCTSConfig(
        num_simulations=150,
        max_depth=2,
        seed=42,
        use_hybrid_expansion=False,
    )

    battle.available_moves = list(garchomp.moves.values())
    battle.available_switches = []

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

        visualize_turn_by_turn(
            stats['root'],
            max_depth=2,
            score_move_fn=score_move,
            score_switch_fn=score_switch,
            battle=battle,
            ctx_me=ctx_me,
            ctx_opp=ctx_opp
        )

    except Exception as e:
        print(f"\n❌ MCTS Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_switch_in_damage()
    # test_simple_ko_scenario()
