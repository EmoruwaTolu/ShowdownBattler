#!/usr/bin/env python3
"""
Status Move Scoring Test Suite

Tests various battle scenarios to validate status move scoring logic.
"""

import sys
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from unittest.mock import Mock, MagicMock

# Add the parent directory to the path so we can import bot modules
# This makes it work regardless of where the script is run from
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print(f"Added to path: {parent_dir}")
print(f"Current working directory: {os.getcwd()}")

# Now import after fixing the path
try:
    from poke_env.battle import Move
    from poke_env.battle import MoveCategory
    from poke_env.battle import Pokemon
    from poke_env.battle import PokemonType
    from poke_env.battle import Status
    print("✅ Successfully imported poke_env modules (0.11.0)")
except ImportError as e:
    print(f"❌ Error importing poke_env: {e}")
    print("Make sure poke-env is installed: pip install poke-env")
    sys.exit(1)

try:
    from bot.scoring.status_score import score_status_move
    from bot.model.ctx import EvalContext
    print("✅ Successfully imported bot modules")
except ImportError as e:
    print(f"❌ Error importing bot modules: {e}")
    print(f"Current sys.path: {sys.path}")
    print("\nMake sure you're running from the battler directory:")
    print("  cd C:\\Users\\emoru\\OneDrive\\Documents\\battler")
    print("  python testing/test_status_scenarios.py")
    sys.exit(1)


@dataclass
class ScenarioResult:
    """Results from testing a scenario."""
    scenario_name: str
    move_scores: Dict[str, float]
    winner: str
    expected_winner: str
    passed: bool
    details: str


class StatusMoveTestSuite:
    """Test suite for status move scoring."""
    
    def __init__(self):
        self.results: List[ScenarioResult] = []
    
    def create_mock_pokemon(
        self, 
        species: str,
        hp: float = 1.0,
        max_hp: int = 100,
        level: int = 50,
        types: List[PokemonType] = None,
        stats: Dict[str, int] = None,
        moves: Dict[str, Move] = None,
        ability: str = None,
        item: str = None,
    ) -> Pokemon:
        """Create a mock Pokemon for testing."""
        mon = Mock(spec=Pokemon)
        mon.species = species
        mon.current_hp_fraction = hp
        mon.max_hp = max_hp
        mon.level = level
        
        # Types
        if types:
            mon.type_1 = types[0]
            mon.type_2 = types[1] if len(types) > 1 else None
            mon.types = types  # Add types list for damage calc
            mon.original_types = types  # For Terastal/type changes
        else:
            mon.type_1 = PokemonType.NORMAL
            mon.type_2 = None
            mon.types = [PokemonType.NORMAL]
            mon.original_types = [PokemonType.NORMAL]
        
        # Stats
        if stats:
            mon.stats = stats
        else:
            # Default balanced stats
            mon.stats = {
                'hp': max_hp,
                'atk': 100,
                'def': 100,
                'spa': 100,
                'spd': 100,
                'spe': 100,
            }
        
        # Boosts
        mon.boosts = {
            'atk': 0, 'def': 0, 'spa': 0, 
            'spd': 0, 'spe': 0, 'accuracy': 0, 'evasion': 0
        }
        
        # Moves
        mon.moves = moves or {}
        
        # Other attributes needed by damage calc
        mon.ability = ability
        mon.item = item
        mon.status = None
        mon.effects = {}
        mon.base_species = species
        
        # Add possible_abilities for damage calc
        mon.possible_abilities = [ability] if ability else []
        
        return mon
    
    def create_mock_move(
        self,
        move_id: str,
        category: MoveCategory,
        base_power: int = 0,
        move_type: PokemonType = PokemonType.NORMAL,
        accuracy: float = 1.0,
        status: Status = None,
    ) -> Move:
        """Create a mock move for testing."""
        move = Mock(spec=Move)
        move.id = move_id
        move.category = category
        move.base_power = base_power
        move.type = move_type
        move.accuracy = accuracy
        move.status = status
        
        # Proper entry dict for damage calculator
        move.entry = {
            'flags': {},
            'basePower': base_power,
            'type': move_type.name if hasattr(move_type, 'name') else str(move_type),
            'category': category.name if hasattr(category, 'name') else str(category),
        }
        
        move.breaks_protect = False
        move.priority = 0
        move.target = None
        move.current_pp = 10
        move.max_pp = 10
        move.n_hit = [1, 1]  # Single-hit move by default
        
        return move
    
    def create_mock_battle(
        self,
        active: Pokemon,
        opponent: Pokemon,
        team: Dict[str, Pokemon],
        opponent_team: Dict[str, Pokemon],
    ):
        """Create a mock battle object."""
        battle = Mock()
        battle.active_pokemon = active
        battle.opponent_active_pokemon = opponent
        battle.team = team
        battle.opponent_team = opponent_team
        battle.player_role = "p1"
        battle.opponent_role = "p2"
        
        # Add all active Pokemon for damage calc
        battle.all_active_pokemons = [active, opponent]
        
        battle.fields = {}
        battle.weather = {}
        battle.side_conditions = {}
        battle.opponent_side_conditions = {}
        
        # Mock available_moves (needed by status_score)
        battle.available_moves = []
        battle.available_switches = []
        battle.force_switch = False
        battle.maybe_trapped = False
        battle.trapped = False
        
        # Mock get_pokemon method
        def get_pokemon(identifier: str):
            if identifier in team:
                return team[identifier]
            elif identifier in opponent_team:
                return opponent_team[identifier]
            return None
        
        battle.get_pokemon = get_pokemon
        
        return battle
    
    def test_scenario_sableye_vs_zacian(self) -> ScenarioResult:
        """
        Test: Sableye vs Zacian with Jirachi ally
        
        Expected: Will-O-Wisp wins (saves Jirachi from OHKO)
        """
        print("\n  Creating Pokemon...")
        
        # Create Pokemon
        sableye = self.create_mock_pokemon(
            species="Sableye",
            hp=1.0,
            max_hp=110,
            types=[PokemonType.GHOST, PokemonType.DARK],
            stats={'hp': 110, 'atk': 95, 'def': 95, 'spa': 85, 'spd': 85, 'spe': 70},
            ability="prankster",
        )
        
        zacian = self.create_mock_pokemon(
            species="Zacian",
            hp=1.0,
            max_hp=142,
            types=[PokemonType.STEEL, PokemonType.FAIRY],
            stats={'hp': 142, 'atk': 255, 'def': 135, 'spa': 100, 'spd': 135, 'spe': 168},
            item="rustedsword",
        )
        
        jirachi = self.create_mock_pokemon(
            species="Jirachi",
            hp=1.0,
            max_hp=160,
            types=[PokemonType.STEEL, PokemonType.PSYCHIC],
            stats={'hp': 160, 'atk': 120, 'def': 120, 'spa': 120, 'spd': 120, 'spe': 120},
        )
        
        print("  Creating moves...")
        
        # Create moves
        sableye.moves = {
            'willowisp': self.create_mock_move(
                'willowisp', MoveCategory.STATUS, 
                move_type=PokemonType.FIRE, accuracy=0.85
            ),
            'knockoff': self.create_mock_move(
                'knockoff', MoveCategory.PHYSICAL, base_power=65,
                move_type=PokemonType.DARK
            ),
            'thunderwave': self.create_mock_move(
                'thunderwave', MoveCategory.STATUS,
                move_type=PokemonType.ELECTRIC, accuracy=0.90
            ),
        }
        
        zacian.moves = {
            'closecombat': self.create_mock_move(
                'closecombat', MoveCategory.PHYSICAL, base_power=120,
                move_type=PokemonType.FIGHTING
            ),
            'playrough': self.create_mock_move(
                'playrough', MoveCategory.PHYSICAL, base_power=90,
                move_type=PokemonType.FAIRY
            ),
            'crunch': self.create_mock_move(
                'crunch', MoveCategory.PHYSICAL, base_power=80,
                move_type=PokemonType.DARK
            ),
            'psychicfangs': self.create_mock_move(
                'psychicfangs', MoveCategory.PHYSICAL, base_power=85,
                move_type=PokemonType.PSYCHIC
            ),
        }
        
        jirachi.moves = {
            'ironhead': self.create_mock_move(
                'ironhead', MoveCategory.PHYSICAL, base_power=80,
                move_type=PokemonType.STEEL
            ),
        }
        
        print("  Creating battle...")
        
        # Create battle
        battle = self.create_mock_battle(
            active=sableye,
            opponent=zacian,
            team={
                'p1: Sableye': sableye,
                'p1: Jirachi': jirachi,
            },
            opponent_team={
                'p2: Zacian': zacian,
            }
        )
        
        print("  Creating context...")
        
        # Create context - EvalContext needs the battle object
        ctx = EvalContext(me=sableye, opp=zacian, battle=battle, cache={})
        
        print("  Scoring moves...")
        
        # Score each move
        move_scores = {}
        try:
            for move_name, move in sableye.moves.items():
                print(f"    Scoring {move_name}...")
                score = score_status_move(move, battle, ctx)
                move_scores[move_name] = score
                print(f"    {move_name}: {score:.2f}")
        except Exception as e:
            import traceback
            print(f"\n❌ Error during scoring:")
            traceback.print_exc()
            return ScenarioResult(
                scenario_name="Sableye vs Zacian",
                move_scores={},
                winner="ERROR",
                expected_winner="willowisp",
                passed=False,
                details=f"Error during scoring: {e}"
            )
        
        # Determine winner
        winner = max(move_scores, key=move_scores.get) if move_scores else "NONE"
        expected = "willowisp"
        passed = winner == expected
        
        details = (
            f"Will-O-Wisp should win because it saves Jirachi from OHKO.\n"
            f"  Without burn: Close Combat deals 121-143% to Jirachi (OHKO)\n"
            f"  With burn: Damage halved to 61-71% (2HKO)\n"
            f"  Burn team synergy: ~+55 (saves ally from death)\n"
            f"  Thunder Wave team synergy: ~+28 (speed control only)"
        )
        
        return ScenarioResult(
            scenario_name="Sableye vs Zacian (with Jirachi)",
            move_scores=move_scores,
            winner=winner,
            expected_winner=expected,
            passed=passed,
            details=details
        )
    
    def run_all_tests(self) -> None:
        """Run all test scenarios."""
        print("=" * 80)
        print("STATUS MOVE SCORING TEST SUITE")
        print("=" * 80)
        print()
        
        scenarios = [
            ("Sableye vs Zacian", self.test_scenario_sableye_vs_zacian),
            # Add more scenarios here as needed
        ]
        
        for name, test_func in scenarios:
            print(f"\n{'=' * 80}")
            print(f"TEST: {name}")
            print('=' * 80)
            
            try:
                result = test_func()
                self.results.append(result)
                
                print(f"\nRESULT: {'✅ PASSED' if result.passed else '❌ FAILED'}")
                print(f"Winner: {result.winner} (expected: {result.expected_winner})")
                print(f"\nMove Scores:")
                for move, score in sorted(result.move_scores.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {move:20s}: {score:6.2f}")
                print(f"\nDetails:\n{result.details}")
                
            except Exception as e:
                print(f"\n❌ TEST FAILED WITH EXCEPTION: {e}")
                import traceback
                traceback.print_exc()
        
        # Summary
        print(f"\n{'=' * 80}")
        print("TEST SUMMARY")
        print('=' * 80)
        
        if self.results:
            passed = sum(1 for r in self.results if r.passed)
            total = len(self.results)
            
            print(f"\nPassed: {passed}/{total} ({100*passed/total:.1f}%)")
            print()
            
            for result in self.results:
                status = "✅" if result.passed else "❌"
                print(f"{status} {result.scenario_name}")
        else:
            print("\nNo tests completed!")


def main():
    """Run the test suite."""
    print("\n" + "=" * 80)
    print("Starting Status Move Scoring Tests")
    print("=" * 80)
    
    suite = StatusMoveTestSuite()
    suite.run_all_tests()
    
    print("\n" + "=" * 80)
    print("Tests Complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
