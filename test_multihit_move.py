"""
Test to see how poke_env's damage calculator handles multi-hit moves.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poke_env.data import GenData
from test_random_randbats import load_randbats_data, create_pokemon_from_randbats
from test_real_heuristics import create_mock_battle as _create_mock_battle_orig

def create_mock_battle(*args, **kwargs):
    battle = _create_mock_battle_orig(*args, **kwargs)
    battle.gen = 9
    return battle

print("=" * 80)
print("MULTI-HIT MOVE DAMAGE CALCULATION TEST")
print("=" * 80)

gen_data = GenData.from_gen(9)
randbats_data = load_randbats_data()

# Create Weavile with Triple Axel
weavile, _, _, _, _ = create_pokemon_from_randbats("Weavile", randbats_data["Weavile"], gen_data)
weavile._identifier_string = "p1: Weavile"
weavile.current_hp_fraction = 1.0

# Create Garchomp as target
garchomp, _, _, _, _ = create_pokemon_from_randbats("Garchomp", randbats_data["Garchomp"], gen_data)
garchomp._identifier_string = "p2: Garchomp"
garchomp.current_hp_fraction = 0.45

# Create battle
battle = create_mock_battle(
    active_identifier=weavile._identifier_string,
    active=weavile,
    opponent_identifier=garchomp._identifier_string,
    opponent=garchomp,
    team={"p1: Weavile": weavile},
    opponent_team={"p2: Garchomp": garchomp}
)

# Check Weavile's moves
print("\nWeavile's moves:")
for move_name, move in weavile.moves.items():
    print(f"  {move_name}: {move.base_power} BP")
    if hasattr(move, 'multihit'):
        print(f"    Multihit: {move.multihit}")

# Test damage calculation for Triple Axel
print("\nTesting damage calculation:")
from poke_env.calc.damage_calc_gen9 import calculate_damage

for move_name, move in weavile.moves.items():
    if 'axel' in move_name.lower() or 'iceshard' in move_name.lower():
        try:
            min_dmg, max_dmg = calculate_damage(
                attacker_identifier=weavile._identifier_string,
                defender_identifier=garchomp._identifier_string,
                move=move,
                battle=battle,
                is_critical=False
            )
            
            avg_dmg = (min_dmg + max_dmg) / 2.0
            garchomp_max_hp = garchomp.max_hp
            dmg_pct = (avg_dmg / garchomp_max_hp) * 100
            
            print(f"\n{move.id}:")
            print(f"  Base Power: {move.base_power}")
            print(f"  Min damage: {min_dmg}")
            print(f"  Max damage: {max_dmg}")
            print(f"  Avg damage: {avg_dmg:.1f}")
            print(f"  Garchomp max HP: {garchomp_max_hp}")
            print(f"  Damage %: {dmg_pct:.1f}%")
            print(f"  Garchomp current HP %: {garchomp.current_hp_fraction * 100:.0f}%")
            print(f"  Would KO: {'YES' if avg_dmg >= garchomp.current_hp_fraction * garchomp_max_hp else 'NO'}")
            
        except Exception as e:
            print(f"\n{move.id}: ERROR - {e}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print("Check if Triple Axel's damage accounts for all 3 hits.")
print("Triple Axel should do ~3x the damage of a single 20 BP move.")
print("If the damage is only showing 20 BP worth, we need to fix the calc.")