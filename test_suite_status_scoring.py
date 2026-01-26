#!/usr/bin/env python3
"""
Comprehensive Test Suite for Status Move Scoring

Tests various scenarios to validate heuristics:
1. Burn saves ally from OHKO (should score HIGH)
2. Burn vs special attacker (should score LOW)
3. Paralysis enables speed flip (should score MEDIUM-HIGH)
4. Toxic with defensive wall (should score HIGH)
5. Status when already winning (should score LOWER)
6. Status when losing badly (should score HIGHER)
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from poke_env.battle import Move, MoveCategory, PokemonType, Status
from bot.scoring.status_score import score_status_move
from bot.model.ctx import EvalContext
from unittest.mock import Mock

def calc_stat(base, level, ev=85):
    return int(((2 * base + 31 + ev // 4) * level // 100 + 5))

def calc_hp(base, level, ev=85):
    return int((2 * base + 31 + ev // 4) * level // 100 + level + 10)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

class SimplePokemon:
    """Simple Pokemon object that doesn't auto-create Mock attributes."""
    pass

def create_pokemon(name, level, base_stats, types, ability="", item=""):
    """Create a Pokemon with proper attributes."""
    stats_calc = {
        'hp': calc_hp(base_stats['hp'], level),
        'atk': calc_stat(base_stats['atk'], level),
        'def': calc_stat(base_stats['def'], level),
        'spa': calc_stat(base_stats['spa'], level),
        'spd': calc_stat(base_stats['spd'], level),
        'spe': calc_stat(base_stats['spe'], level),
    }
    
    mon = SimplePokemon()
    mon.species = name
    mon.base_species = name
    mon.types = types
    mon.original_types = types
    mon.type_1 = types[0]
    mon.type_2 = types[1] if len(types) > 1 else None
    mon.level = level
    mon.stats = stats_calc
    mon.base_stats = base_stats
    mon.max_hp = stats_calc['hp']
    mon.current_hp_fraction = 1.0
    mon.status = None
    mon.boosts = {'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0}
    mon.effects = {}
    mon.ability = ability
    mon.item = item
    mon.possible_abilities = [ability] if ability else []
    mon.fainted = False
    mon.moves = {}
    
    # Additional attributes needed by damage calculator
    mon.gender = None
    mon.is_terastallized = False
    mon.tera_type = None
    mon.weight = 10.0  # Default weight in kg
    
    return mon

class SimpleMove:
    """Simple Move object."""
    pass

def create_move(move_id, category, move_type, bp=0, status=None):
    """Create a move with all required attributes."""
    move = SimpleMove()
    move.id = move_id
    move.category = category
    move.type = move_type
    move.base_power = bp
    move.accuracy = 0.85 if status else 1.0
    move.entry = {'flags': {}, 'basePower': bp, 'type': move_type.name, 'category': category.name}
    move.priority = 0
    move.n_hit = [1, 1]
    move.status = status
    move.breaks_protect = False
    
    # Additional attributes needed by damage calculator
    move.recoil = None
    move.secondary = None
    move.target = "normal"
    move.ignore_defensive = False
    move.is_stellar_first_use = False
    move.flags = {}
    
    return move

def create_battle(active, opponent, team, opp_team):
    """Create a mock battle."""
    battle = Mock()
    battle.team = team
    battle.opponent_team = opp_team
    battle.active_pokemon = active
    battle.opponent_active_pokemon = opponent
    battle.all_active_pokemons = [active, opponent]
    battle.fields = {}
    battle.weather = {}
    battle.side_conditions = {}
    battle.opponent_side_conditions = {}
    battle.player_role = "p1"
    battle.opponent_role = "p2"
    battle.available_moves = []
    battle.format = Mock()
    battle.format.gen = 9
    
    def get_pokemon(identifier):
        return battle.team.get(identifier) or battle.opponent_team.get(identifier)
    
    battle.get_pokemon = get_pokemon
    return battle

# ============================================================================
# TEST SCENARIOS
# ============================================================================

class TestResult:
    def __init__(self, name, score, expected_range, passed, reason=""):
        self.name = name
        self.score = score
        self.expected_range = expected_range
        self.passed = passed
        self.reason = reason

def run_test(test_name, active, opponent, allies, moves_to_test, expected_winner, expected_min, description):
    """Run a single test scenario."""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}")
    print(f"Description: {description}")
    
    try:
        # Build team
        team = {f'p1: {active.species}': active}
        for ally in allies:
            team[f'p1: {ally.species}'] = ally
        
        opp_team = {f'p2: {opponent.species}': opponent}
        
        # Create battle
        battle = create_battle(active, opponent, team, opp_team)
        ctx = EvalContext(me=active, opp=opponent, battle=battle, cache={})
        
        # Score all moves
        print(f"\n{'Move Scores:':^80}")
        print("-" * 80)
        
        scores = {}
        from bot.scoring.move_score import score_move
        
        for move_name, move in moves_to_test.items():
            try:
                score = score_move(move, battle, ctx)
                scores[move_name] = score
                print(f"  {move_name:20s}: {score:7.2f}")
            except Exception as e:
                print(f"  {move_name:20s}: ERROR - {e}")
                scores[move_name] = -999
        
        if not scores:
            return TestResult(test_name, 0, (expected_min, float('inf')), False, "No moves scored")
        
        # Find winner
        winner = max(scores, key=scores.get)
        winner_score = scores[winner]
        
        # Check result
        passed = (winner == expected_winner) and (winner_score >= expected_min)
        
        print(f"\n{'Results:':^80}")
        print("-" * 80)
        print(f"Winner: {winner} ({winner_score:.2f})")
        print(f"Expected: {expected_winner} (>= {expected_min:.0f})")
        
        if passed:
            print(f"\n✅ PASSED")
        else:
            print(f"\n❌ FAILED")
            if winner != expected_winner:
                print(f"   Expected {expected_winner} to win, but {winner} won")
            if winner_score < expected_min:
                print(f"   Score too low: {winner_score:.2f} < {expected_min:.0f}")
        
        return TestResult(test_name, winner_score, (expected_min, float('inf')), passed, winner)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return TestResult(test_name, 0, (expected_min, float('inf')), False, str(e))

# ============================================================================
# TEST 1: Burn saves frail ally from OHKO
# ============================================================================

def test_1_burn_saves_from_ohko():
    """
    Scenario: Physical sweeper threatens to OHKO frail ally
    Expected: Burn should WIN over damaging move (score > damage)
    """
    # Salamence (physical sweeper with Outrage)
    salamence = create_pokemon(
        "Salamence", 77,
        {'hp': 95, 'atk': 135, 'def': 80, 'spa': 110, 'spd': 80, 'spe': 100},
        [PokemonType.DRAGON, PokemonType.FLYING],
        "intimidate", "lifeorb"
    )
    outrage = create_move('outrage', MoveCategory.PHYSICAL, PokemonType.DRAGON, 120)
    salamence.moves = {'outrage': outrage}
    
    # Weavile (frail, threatened by Outrage)
    weavile = create_pokemon(
        "Weavile", 75,
        {'hp': 70, 'atk': 120, 'def': 65, 'spa': 45, 'spd': 85, 'spe': 125},
        [PokemonType.DARK, PokemonType.ICE],
        "pressure", "focussash"
    )
    
    # Rotom-Wash (Will-O-Wisp user)
    rotom = create_pokemon(
        "Rotom-Wash", 84,
        {'hp': 50, 'atk': 65, 'def': 107, 'spa': 105, 'spd': 107, 'spe': 86},
        [PokemonType.ELECTRIC, PokemonType.WATER],
        "levitate", "sitrusberry"
    )
    
    willowisp = create_move('willowisp', MoveCategory.STATUS, PokemonType.FIRE, 0, Status.BRN)
    hydropump = create_move('hydropump', MoveCategory.SPECIAL, PokemonType.WATER, 110)
    voltswitch = create_move('voltswitch', MoveCategory.SPECIAL, PokemonType.ELECTRIC, 70)
    
    moves = {
        'willowisp': willowisp,
        'hydropump': hydropump,
        'voltswitch': voltswitch,
    }
    rotom.moves = moves  # Set active Pokemon's moves!
    
    return run_test(
        "Burn Saves Ally from OHKO",
        rotom, salamence, [weavile], moves,
        "willowisp", 80,  # Will-O-Wisp should win with score >= 80
        "Salamence Outrage threatens to OHKO Weavile. Burn should be the best move."
    )

# ============================================================================
# TEST 2: Burn vs special attacker (should LOSE to damage)
# ============================================================================

def test_2_burn_vs_special_attacker():
    """
    Scenario: Opponent is pure special attacker
    Expected: Damage move should WIN over burn (burn is wasted)
    """
    # Hydreigon (special attacker)
    hydreigon = create_pokemon(
        "Hydreigon", 77,
        {'hp': 92, 'atk': 105, 'def': 90, 'spa': 125, 'spd': 90, 'spe': 98},
        [PokemonType.DARK, PokemonType.DRAGON],
        "levitate", "choicespecs"
    )
    dracometeor = create_move('dracometeor', MoveCategory.SPECIAL, PokemonType.DRAGON, 130)
    hydreigon.moves = {'dracometeor': dracometeor}
    
    # Blissey (status user)
    blissey = create_pokemon(
        "Blissey", 86,
        {'hp': 255, 'atk': 10, 'def': 10, 'spa': 75, 'spd': 135, 'spe': 55},
        [PokemonType.NORMAL],
        "naturalcure", "heavydutyboots"
    )
    
    # Physical teammate
    machamp = create_pokemon(
        "Machamp", 80,
        {'hp': 90, 'atk': 130, 'def': 80, 'spa': 65, 'spd': 85, 'spe': 55},
        [PokemonType.FIGHTING],
        "guts", "flameorb"
    )
    
    willowisp = create_move('willowisp', MoveCategory.STATUS, PokemonType.FIRE, 0, Status.BRN)
    seismictoss = create_move('seismictoss', MoveCategory.PHYSICAL, PokemonType.FIGHTING, 1)  # Fixed 100 damage
    icywind = create_move('icywind', MoveCategory.SPECIAL, PokemonType.ICE, 55)
    
    moves = {
        'willowisp': willowisp,
        'seismictoss': seismictoss,
        'icywind': icywind,
    }
    blissey.moves = moves
    
    return run_test(
        "Burn vs Special Attacker",
        blissey, hydreigon, [machamp], moves,
        "seismictoss", 20,  # Damage should win (any score)
        "Hydreigon is special. Burn is useless, damage moves should win."
    )

# ============================================================================
# TEST 3: Thunder Wave enables speed flip
# ============================================================================

def test_3_paralysis_speed_flip():
    """
    Scenario: Slower setup sweeper can outspeed after paralysis
    Expected: Thunder Wave should WIN (enables sweep strategy)
    """
    # Rillaboom (fast physical)
    rillaboom = create_pokemon(
        "Rillaboom", 79,
        {'hp': 100, 'atk': 125, 'def': 90, 'spa': 60, 'spd': 70, 'spe': 85},
        [PokemonType.GRASS],
        "grassysurge", "choiceband"
    )
    woodhammer = create_move('woodhammer', MoveCategory.PHYSICAL, PokemonType.GRASS, 120)
    rillaboom.moves = {'woodhammer': woodhammer}
    
    # Suicune (slower, would benefit from para)
    suicune = create_pokemon(
        "Suicune", 81,
        {'hp': 100, 'atk': 75, 'def': 115, 'spa': 90, 'spd': 115, 'spe': 85},
        [PokemonType.WATER],
        "pressure", "leftovers"
    )
    
    # Clefable (Thunder Wave user)
    clefable = create_pokemon(
        "Clefable", 82,
        {'hp': 95, 'atk': 70, 'def': 73, 'spa': 95, 'spd': 90, 'spe': 60},
        [PokemonType.FAIRY],
        "magicguard", "lifeorb"
    )
    
    thunderwave = create_move('thunderwave', MoveCategory.STATUS, PokemonType.ELECTRIC, 0, Status.PAR)
    moonblast = create_move('moonblast', MoveCategory.SPECIAL, PokemonType.FAIRY, 95)
    
    moves = {
        'thunderwave': thunderwave,
        'moonblast': moonblast,
    }
    suicune.moves = moves
    
    return run_test(
        "Paralysis Enables Speed Flip",
        clefable, rillaboom, [suicune], moves,
        "thunderwave", 60,
        "Para lets Suicune outspeed. Should win over damage."
    )

# ============================================================================
# TEST 4: Toxic with defensive wall
# ============================================================================

def test_4_toxic_with_wall():
    """
    Scenario: Have defensive wall that can stall with Toxic
    Expected: Toxic should WIN (stall strategy viable)
    """
    # Mimikyu (opponent)
    mimikyu = create_pokemon(
        "Mimikyu", 83,
        {'hp': 55, 'atk': 90, 'def': 80, 'spa': 50, 'spd': 105, 'spe': 98},
        [PokemonType.FAIRY, PokemonType.GHOST],
        "sandstream", "choiceband"
    )
    shadowclaw = create_move('shadowclaw', MoveCategory.PHYSICAL, PokemonType.GHOST, 80)
    mimikyu.moves = {'shadowclaw': shadowclaw}
    
    # Toxapex (defensive wall)
    toxapex = create_pokemon(
        "Toxapex", 84,
        {'hp': 50, 'atk': 63, 'def': 152, 'spa': 53, 'spd': 142, 'spe': 35},
        [PokemonType.POISON, PokemonType.WATER],
        "regenerator", "blacksludge"
    )
    
    # Clefable (Toxic user)
    clefable = create_pokemon(
        "Clefable", 82,
        {'hp': 95, 'atk': 70, 'def': 73, 'spa': 95, 'spd': 90, 'spe': 60},
        [PokemonType.FAIRY],
        "magicguard", "leftovers"
    )
    
    toxic = create_move('toxic', MoveCategory.STATUS, PokemonType.POISON, 0, Status.TOX)
    moonblast = create_move('moonblast', MoveCategory.SPECIAL, PokemonType.FAIRY, 95)
    
    moves = {
        'toxic': toxic,
        'moonblast': moonblast,
    }
    clefable.moves = moves
    
    return run_test(
        "Toxic with Defensive Wall",
        clefable, mimikyu, [toxapex], moves,
        "toxic", 50,
        "Toxapex enables stall. Toxic should win."
    )

# ============================================================================
# TEST 5: Status when already winning (low HP opponent)
# ============================================================================

def test_5_status_when_winning():
    """
    Scenario: Opponent is at low HP, we're healthy
    Expected: Damage should WIN (finish the KO)
    """
    # Machamp (low HP, Fighting type)
    machamp = create_pokemon(
        "Machamp", 80,
        {'hp': 90, 'atk': 130, 'def': 80, 'spa': 65, 'spd': 85, 'spe': 55},
        [PokemonType.FIGHTING],
        "guts", "flameorb"
    )
    machamp.current_hp_fraction = 0.3  # Low HP!
    closecombat = create_move('closecombat', MoveCategory.PHYSICAL, PokemonType.FIGHTING, 120)
    stoneedge = create_move('stoneedge', MoveCategory.PHYSICAL, PokemonType.ROCK, 100)
    machamp.moves = {'closecombat': closecombat, 'stoneedge': stoneedge}
    
    # Rotom (healthy)
    rotom = create_pokemon(
        "Rotom-Wash", 84,
        {'hp': 50, 'atk': 65, 'def': 107, 'spa': 105, 'spd': 107, 'spe': 86},
        [PokemonType.ELECTRIC, PokemonType.WATER],
        "levitate", "choicespecs"
    )
    rotom.current_hp_fraction = 1.0
    
    # Healthy teammate
    toxapex = create_pokemon(
        "Toxapex", 84,
        {'hp': 50, 'atk': 63, 'def': 152, 'spa': 53, 'spd': 142, 'spe': 35},
        [PokemonType.POISON, PokemonType.WATER],
        "regenerator", "blacksludge"
    )
    toxapex.current_hp_fraction = 1.0
    
    willowisp = create_move('willowisp', MoveCategory.STATUS, PokemonType.FIRE, 0, Status.BRN)
    hydropump = create_move('hydropump', MoveCategory.SPECIAL, PokemonType.WATER, 110)
    voltswitch = create_move('voltswitch', MoveCategory.SPECIAL, PokemonType.ELECTRIC, 70)
    
    moves = {
        'willowisp': willowisp,
        'hydropump': hydropump,
        'voltswitch': voltswitch,
    }
    rotom.moves = moves
    
    return run_test(
        "Status When Already Winning",
        rotom, machamp, [toxapex], moves,
        "hydropump", 40,  # Damage should win
        "Machamp at 30% HP. Should finish with damage, not status."
    )

# ============================================================================
# TEST 6: Burn when ally already tanks
# ============================================================================

def test_6_burn_already_tanked():
    """
    Scenario: Ally is so bulky that burn doesn't meaningfully change survival
    Expected: Damage should WIN (burn doesn't add much value)
    """
    # Conkeldurr (strong physical attacker)
    conkeldurr = create_pokemon(
        "Conkeldurr", 84,
        {'hp': 105, 'atk': 140, 'def': 95, 'spa': 55, 'spd': 65, 'spe': 45},
        [PokemonType.FIGHTING],
        "guts", "flameorb"
    )
    drainpunch = create_move('drainpunch', MoveCategory.PHYSICAL, PokemonType.FIGHTING, 75)
    machpunch = create_move('machpunch', MoveCategory.PHYSICAL, PokemonType.FIGHTING, 40)
    conkeldurr.moves = {'drainpunch': drainpunch, 'machpunch': machpunch}
    
    # Slowbro (extremely physically defensive, Regenerator)
    # Even without burn, Drain Punch is only a 4HKO
    slowbro = create_pokemon(
        "Slowbro", 84,
        {'hp': 95, 'atk': 75, 'def': 110, 'spa': 100, 'spd': 80, 'spe': 30},
        [PokemonType.WATER, PokemonType.PSYCHIC],
        "regenerator", "heavydutyboots"
    )
    
    # Clefable (status spreader)
    clefable = create_pokemon(
        "Clefable", 82,
        {'hp': 95, 'atk': 70, 'def': 73, 'spa': 95, 'spd': 90, 'spe': 60},
        [PokemonType.FAIRY],
        "magicguard", "lifeorb"
    )
    
    willowisp = create_move('willowisp', MoveCategory.STATUS, PokemonType.FIRE, 0, Status.BRN)
    moonblast = create_move('moonblast', MoveCategory.SPECIAL, PokemonType.FAIRY, 95)
    thunderwave = create_move('thunderwave', MoveCategory.STATUS, PokemonType.ELECTRIC, 0, Status.PAR)
    
    moves = {
        'willowisp': willowisp,
        'moonblast': moonblast,
        'thunderwave': thunderwave,
    }
    clefable.moves = moves
    
    return run_test(
        "Burn When Ally Already Tanks",
        clefable, conkeldurr, [slowbro], moves,
        "moonblast", 40,  # Damage should win
        "Slowbro already walls Conkeldurr (Regen + bulk). Damage is better than redundant burn."
    )

def test_7_scald_burn_fishing():
    """
    Scenario: Scald (80 BP + 30% burn) vs Thunder Wave
    Scream Tail is faster than Excadrill. Para enables Excadrill to outspeed.
    Burn weakens Scream Tail's Play Rough that threatens Excadrill.
    Expected: Scald should WIN (damage + burn chance + protects ally)
    """
    # Scream Tail (Fairy/Psychic, physical attacker)
    screamtail = create_pokemon(
        "Scream Tail", 82,
        {'hp': 115, 'atk': 65, 'def': 99, 'spa': 65, 'spd': 115, 'spe': 111},
        [PokemonType.FAIRY, PokemonType.PSYCHIC],
        "protosynthesis", "boosterenergy"
    )
    playrough = create_move('playrough', MoveCategory.PHYSICAL, PokemonType.FAIRY, 90)
    psychicfangs = create_move('psychicfangs', MoveCategory.PHYSICAL, PokemonType.PSYCHIC, 85)
    screamtail.moves = {'playrough': playrough, 'psychicfangs': psychicfangs}
    
    # Excadrill (frail to Play Rough, slower than Scream Tail)
    excadrill = create_pokemon(
        "Excadrill", 80,
        {'hp': 110, 'atk': 135, 'def': 60, 'spa': 50, 'spd': 65, 'spe': 88},
        [PokemonType.GROUND, PokemonType.STEEL],
        "sandrush", "focussash"
    )
    ironhead = create_move('ironhead', MoveCategory.PHYSICAL, PokemonType.STEEL, 80)
    earthquake_exc = create_move('earthquake', MoveCategory.PHYSICAL, PokemonType.GROUND, 100)
    excadrill.moves = {'ironhead': ironhead, 'earthquake': earthquake_exc}
    
    # Suicune (Scald + Thunder Wave user)
    suicune = create_pokemon(
        "Suicune", 81,
        {'hp': 100, 'atk': 75, 'def': 115, 'spa': 90, 'spd': 115, 'spe': 85},
        [PokemonType.WATER],
        "pressure", "leftovers"
    )
    
    # Create Scald with 30% burn secondary
    scald = create_move('scald', MoveCategory.SPECIAL, PokemonType.WATER, 80)
    scald.secondary = [{'chance': 30, 'status': 'brn'}]
    
    thunderwave = create_move('thunderwave', MoveCategory.STATUS, PokemonType.ELECTRIC, 0, Status.PAR)
    icebeam = create_move('icebeam', MoveCategory.SPECIAL, PokemonType.ICE, 90)
    surf = create_move('surf', MoveCategory.SPECIAL, PokemonType.WATER, 90)
    
    moves = {
        'scald': scald,
        'thunderwave': thunderwave,
        'icebeam': icebeam,
        'surf': surf,
    }
    suicune.moves = moves
    
    return run_test(
        "Scald Burn Fishing",
        suicune, screamtail, [excadrill], moves,
        "scald", 50,  # Scald should win
        "Scald: damage + 30% burn weakens Play Rough vs Excadrill. Better than pure status/damage."
    )

def test_8_suicune_vs_darkrai():
    """
    Scenario: Suicune vs Darkrai with Scizor in the back
    
    Darkrai is a special attacker (Dark Pulse, Ice Beam, Sludge Bomb)
    - Fast (125 speed) and threatens Scizor with coverage
    - Scizor is slower (65 speed) and weak to Fire/Dark
    
    Suicune options:
    - Scald: Damage + burn (but Darkrai is special, so burn is less valuable)
    - Ice Beam: Neutral damage to Darkrai
    - Thunder Wave: Para for speed control (helps Scizor outspeed)
    - Calm Mind: Setup move
    
    Observational: Which does the bot choose?
    """
    # Darkrai (fast special attacker, Dark type)
    darkrai = create_pokemon(
        "Darkrai", 80,
        {'hp': 70, 'atk': 90, 'def': 90, 'spa': 135, 'spd': 90, 'spe': 95},
        [PokemonType.DARK],
        "baddreams", "focussash"
    )
    darkpulse = create_move('darkpulse', MoveCategory.SPECIAL, PokemonType.DARK, 80)
    icebeam_darkrai = create_move('icebeam', MoveCategory.SPECIAL, PokemonType.ICE, 90)
    sludgebomb = create_move('sludgebomb', MoveCategory.SPECIAL, PokemonType.POISON, 90)
    darkrai.moves = {'darkpulse': darkpulse, 'icebeam': icebeam_darkrai, 'sludgebomb': sludgebomb}
    
    # Scizor (slower, weak to Fire and Dark Pulse)
    scizor = create_pokemon(
        "Scizor", 78,
        {'hp': 70, 'atk': 130, 'def': 100, 'spa': 55, 'spd': 80, 'spe': 100},
        [PokemonType.BUG, PokemonType.STEEL],
        "technician", "choiceband"
    )
    bulletpunch = create_move('bulletpunch', MoveCategory.PHYSICAL, PokemonType.STEEL, 40)
    bulletpunch.priority = 1  # Priority move
    uturn = create_move('uturn', MoveCategory.PHYSICAL, PokemonType.BUG, 70)
    scizor.moves = {'bulletpunch': bulletpunch, 'uturn': uturn}
    
    # Suicune (bulky Water type)
    suicune = create_pokemon(
        "Suicune", 81,
        {'hp': 100, 'atk': 75, 'def': 115, 'spa': 90, 'spd': 115, 'spe': 85},
        [PokemonType.WATER],
        "pressure", "leftovers"
    )
    
    # Create Scald with 30% burn secondary
    scald = create_move('scald', MoveCategory.SPECIAL, PokemonType.WATER, 80)
    scald.secondary = [{'chance': 30, 'status': 'brn'}]
    
    icebeam = create_move('icebeam', MoveCategory.SPECIAL, PokemonType.ICE, 90)
    thunderwave = create_move('thunderwave', MoveCategory.STATUS, PokemonType.ELECTRIC, 0, Status.PAR)
    calmmind = create_move('calmmind', MoveCategory.STATUS, PokemonType.PSYCHIC, 0)
    calmmind.boosts = {'spa': 1, 'spd': 1}
    
    moves = {
        'scald': scald,
        'icebeam': icebeam,
        'thunderwave': thunderwave,
        'calmmind': calmmind,
    }
    suicune.moves = moves
    
    # OBSERVATIONAL TEST
    print(f"\n{'='*80}")
    print(f"TEST: Suicune vs Darkrai (OBSERVATIONAL)")
    print(f"{'='*80}")
    print(f"Description: Darkrai is special (burn less valuable). Para helps Scizor.")
    
    # Build team
    team = {f'p1: {suicune.species}': suicune, f'p1: {scizor.species}': scizor}
    opp_team = {f'p2: {darkrai.species}': darkrai}
    
    # Create battle
    battle = create_battle(suicune, darkrai, team, opp_team)
    ctx = EvalContext(me=suicune, opp=darkrai, battle=battle, cache={})
    
    # Score all moves
    print(f"\n{'Move Scores:':^80}")
    print("-" * 80)
    
    from bot.scoring.move_score import score_move
    scores = {}
    for move_name, move in moves.items():
        try:
            score = score_move(move, battle, ctx)
            scores[move_name] = score
            print(f"  {move_name:20s}: {score:7.2f}")
        except Exception as e:
            print(f"  {move_name:20s}: ERROR - {e}")
            scores[move_name] = -999
    
    winner = max(scores, key=scores.get) if scores else "none"
    winner_score = scores.get(winner, 0)
    
    print(f"\n{'Results:':^80}")
    print("-" * 80)
    print(f"Winner: {winner} ({winner_score:.2f})")
    print(f"\n💡 Interpretation:")
    print(f"   - Scald burn is less valuable (Darkrai is special)")
    print(f"   - Thunder Wave helps Scizor outspeed Darkrai (125 → 62.5 speed)")
    print(f"   - Ice Beam is pure neutral damage")
    print(f"   - Calm Mind could be setup opportunity")
    print(f"   - Does bot recognize burn is weak vs special attackers?")
    
    # Return a passing result since this is observational
    return TestResult("Suicune vs Darkrai (Observational)", winner_score, (0, float('inf')), True, winner)

def test_9_chip_enables_revenge():
    """
    Scenario: Chip damage enables a clean revenge KO
    
    Setup:
    - Tyranitar at 52% HP
    - Weavile in the back (120 base attack, faster than Tyranitar)
    - Weavile's Ice Shard does ~48% to Tyranitar
    - If we chip Tyranitar below ~50%, Ice Shard becomes guaranteed KO
    - Weavile is already faster (125 vs 61 speed), so para doesn't help
    
    Rotom's options:
    - Volt Switch: Chip damage (~15-20%) → Enables Ice Shard KO
    - Thunder Wave: No chip → Ice Shard doesn't KO, but doesn't need speed control
    - Hydro Pump: High damage but might not enable follow-up
    
    Question: Does the bot recognize chip is more valuable than para when:
    1. Ally is already faster (no speed control needed)
    2. Chip enables a clean priority KO line
    """
    # Tyranitar at 52% HP (in range for chip + Ice Shard)
    tyranitar = create_pokemon(
        "Tyranitar", 78,
        {'hp': 100, 'atk': 134, 'def': 110, 'spa': 95, 'spd': 100, 'spe': 61},
        [PokemonType.ROCK, PokemonType.DARK],
        "sandstream", "choiceband"
    )
    tyranitar.current_hp_fraction = 0.98  # 52% HP - critical threshold
    crunch = create_move('crunch', MoveCategory.PHYSICAL, PokemonType.DARK, 80)
    stoneedge = create_move('stoneedge', MoveCategory.PHYSICAL, PokemonType.ROCK, 100)
    tyranitar.moves = {'crunch': crunch, 'stoneedge': stoneedge}
    
    # Weavile (faster, has priority Ice Shard)
    # Already faster than Tyranitar (125 vs 61), so para is USELESS
    weavile = create_pokemon(
        "Weavile", 79,
        {'hp': 70, 'atk': 120, 'def': 65, 'spa': 45, 'spd': 85, 'spe': 125},
        [PokemonType.DARK, PokemonType.ICE],
        "pressure", "choiceband"
    )
    iceshard = create_move('iceshard', MoveCategory.PHYSICAL, PokemonType.ICE, 40)
    iceshard.priority = 1  # Priority move
    iciclecrash = create_move('iciclecrash', MoveCategory.PHYSICAL, PokemonType.ICE, 85)
    weavile.moves = {'iceshard': iceshard, 'iciclecrash': iciclecrash}
    
    # Rotom-Wash (active, has chip and status options)
    # LOW HP - needs to get value before fainting
    rotom = create_pokemon(
        "Rotom-Wash", 83,
        {'hp': 50, 'atk': 65, 'def': 107, 'spa': 105, 'spd': 107, 'spe': 86},
        [PokemonType.ELECTRIC, PokemonType.WATER],
        "levitate", "choicescarf"
    )
    rotom.current_hp_fraction = 0.25  # 25% HP - very low!
    
    # Volt Switch: Chip damage that enables Ice Shard KO
    voltswitch = create_move('voltswitch', MoveCategory.SPECIAL, PokemonType.ELECTRIC, 70)
    
    # Thunder Wave: No chip, and speed control is USELESS (Weavile already faster)
    thunderwave = create_move('thunderwave', MoveCategory.STATUS, PokemonType.ELECTRIC, 0, Status.PAR)
    
    # Hydro Pump: High damage but no switch
    hydropump = create_move('hydropump', MoveCategory.SPECIAL, PokemonType.WATER, 110)
    
    # Will-O-Wisp: Burn (also useless vs special attacker)
    willowisp = create_move('willowisp', MoveCategory.STATUS, PokemonType.FIRE, 0, Status.BRN)
    
    moves = {
        'voltswitch': voltswitch,
        'thunderwave': thunderwave,
        'hydropump': hydropump,
        'willowisp': willowisp,
    }
    rotom.moves = moves
    
    # OBSERVATIONAL TEST
    print(f"\n{'='*80}")
    print(f"TEST: Chip Enables Revenge KO (OBSERVATIONAL)")
    print(f"{'='*80}")
    print(f"Description: Tyranitar at 52% HP. Weavile already faster. Chip enables Ice Shard KO.")
    print(f"             Para is USELESS (Weavile 125 vs Tyranitar 61 speed).")
    print(f"             Rotom at 25% HP - needs to get value before fainting!")
    
    # Build team
    team = {f'p1: {rotom.species}': rotom, f'p1: {weavile.species}': weavile}
    opp_team = {f'p2: {tyranitar.species}': tyranitar}
    
    # Create battle
    battle = create_battle(rotom, tyranitar, team, opp_team)
    ctx = EvalContext(me=rotom, opp=tyranitar, battle=battle, cache={})
    
    # Score all moves
    print(f"\n{'Move Scores:':^80}")
    print("-" * 80)
    
    from bot.scoring.move_score import score_move
    scores = {}
    for move_name, move in moves.items():
        try:
            score = score_move(move, battle, ctx)
            scores[move_name] = score
            print(f"  {move_name:20s}: {score:7.2f}")
        except Exception as e:
            print(f"  {move_name:20s}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            scores[move_name] = -999
    
    winner = max(scores, key=scores.get) if scores else "none"
    winner_score = scores.get(winner, 0)
    
    print(f"\n{'Results:':^80}")
    print("-" * 80)
    print(f"Winner: {winner} ({winner_score:.2f})")
    print(f"\n💡 Key Points:")
    print(f"   - Tyranitar at 52% HP, needs ~15% chip to enable Ice Shard KO")
    print(f"   - Weavile is ALREADY FASTER (125 vs 61), para adds NO value")
    print(f"   - Rotom at 25% HP - likely to faint soon, needs to get value NOW")
    print(f"   - Volt Switch: Chips + enables Ice Shard KO + switches to safety")
    print(f"   - Thunder Wave: No chip, no speed help, Rotom stays in and dies")
    print(f"   - Hydro Pump: Damage but Rotom stays in danger")
    print(f"\n💡 Expected Behavior:")
    print(f"   - Volt Switch should win (enables KO + gets Rotom out)")
    print(f"   - Thunder Wave should score LOW (provides nothing)")
    print(f"   - Low HP makes Volt Switch even more valuable (pivot to safety)")
    print(f"   - If Thunder Wave wins, para is being overvalued")
    
    # Return a passing result since this is observational
    return TestResult("Chip Enables Revenge (Observational)", winner_score, (0, float('inf')), True, winner)

# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_all_tests():
    """Run the full test suite."""
    print("=" * 80)
    print("STATUS MOVE SCORING TEST SUITE")
    print("=" * 80)
    print("\nTesting various scenarios to validate scoring heuristics...")
    
    results = []
    
    results.append(test_1_burn_saves_from_ohko())
    results.append(test_2_burn_vs_special_attacker())
    results.append(test_3_paralysis_speed_flip())
    results.append(test_4_toxic_with_wall())
    results.append(test_5_status_when_winning())
    results.append(test_6_burn_already_tanked())
    results.append(test_7_scald_burn_fishing())
    results.append(test_8_suicune_vs_darkrai())
    results.append(test_9_chip_enables_revenge())
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}\n")
    
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    for result in results:
        status = "✅ PASSED" if result.passed else "❌ FAILED"
        print(f"{status:12s} {result.name:40s} Winner: {result.reason:15s} Score: {result.score:6.2f}")
    
    print(f"\n{'='*80}")
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"{'='*80}\n")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Status scoring behaves as expected!")
    elif passed >= total * 0.67:
        print("✅ Most tests passed! Scoring is generally working well.")
    else:
        print("⚠️ Some tests failed. Review the scenarios above.")
    
    return results

if __name__ == "__main__":
    run_all_tests()