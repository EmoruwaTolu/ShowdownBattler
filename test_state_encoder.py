"""
Tests for bot/learning/state_encoder.py

Run:
    python test_state_encoder.py
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from unittest.mock import Mock

from poke_env.battle import MoveCategory, PokemonType, Status, Effect
from bot.learning.state_encoder import (
    encode_state,
    encode_state_flat,
    N_MON_FEATURES,
    N_FIELD_FEATURES,
    N_ACTIVE_PAIR_FEATURES,
    N_TOTAL,
    MON_FEATURE_NAMES,
    FIELD_FEATURE_NAMES,
    ACTIVE_PAIR_FEATURE_NAMES,
)

# ---------------------------------------------------------------------------
# Minimal mock helpers
# ---------------------------------------------------------------------------

def _mon(
    species="bulbasaur",
    hp_frac=1.0,
    status=None,
    types=(PokemonType.GRASS,),
    moves=None,
    boosts=None,
    spe=80,
    effects=None,
    is_terastallized=False,
):
    m = Mock()
    m.species = species
    m.types = list(types)
    m.current_hp_fraction = hp_frac
    m.current_hp = int(100 * hp_frac)
    m.max_hp = 100
    m.status = status
    m.moves = moves or {}
    m.boosts = boosts or {}
    m.stats = {"hp": 100, "atk": 80, "def": 80, "spa": 80, "spd": 80, "spe": spe}
    m.item = None
    m.ability = None
    m.fainted = hp_frac <= 0.0
    m.level = 50
    m.effects = effects or {}
    m.is_terastallized = is_terastallized
    return m


def _move(move_id, bp=80, priority=0, category=MoveCategory.SPECIAL, move_type=PokemonType.FIRE):
    mv = Mock()
    mv.id = move_id
    mv.base_power = bp
    mv.priority = priority
    mv.category = category
    mv.type = move_type
    mv.accuracy = 1.0
    mv.current_pp = mv.max_pp = 16
    return mv


def _make_state(
    my_team=None,
    opp_team=None,
    my_active_idx=0,
    opp_active_idx=0,
    opp_slots=None,
    my_hp=None,
    opp_hp=None,
    my_status=None,
    opp_status=None,
    my_boosts=None,
    opp_boosts=None,
    opp_beliefs=None,
    my_side_conditions=None,
    opp_side_conditions=None,
    shadow_weather=None,
    shadow_fields=None,
    my_volatiles=None,
    opp_volatiles=None,
    used_tera=False,
    opponent_used_tera=False,
):
    if my_team is None:
        my_team = [_mon("charmander"), _mon("squirtle"), _mon("bulbasaur"),
                   _mon("pikachu"),    _mon("mewtwo"),   _mon("snorlax")]
    if opp_team is None:
        opp_team = [_mon("gengar"), _mon("alakazam")]

    my_hp   = my_hp   or {id(p): p.current_hp_fraction for p in my_team}
    opp_hp  = opp_hp  or {id(p): p.current_hp_fraction for p in opp_team}

    state = Mock()

    # Battle mock with Tera tracking
    battle = Mock()
    battle.used_tera = used_tera
    battle.opponent_used_tera = opponent_used_tera
    state.battle = battle

    state.my_team    = my_team
    state.opp_team   = opp_team
    state.my_active  = my_team[my_active_idx] if my_team else None

    if opp_slots is not None:
        state.opp_active = opp_slots[opp_active_idx] if opp_active_idx < len(opp_slots) else None
    else:
        state.opp_active = opp_team[opp_active_idx] if opp_team and opp_active_idx < len(opp_team) else None

    state.my_hp       = my_hp
    state.opp_hp      = opp_hp
    state.my_status   = my_status   or {}
    state.opp_status  = opp_status  or {}
    state.my_boosts   = my_boosts   or {}
    state.opp_boosts  = opp_boosts  or {}
    state.opp_beliefs = opp_beliefs or {}
    state.opp_slots   = opp_slots
    state.opp_active_idx = opp_active_idx
    state.my_side_conditions  = my_side_conditions  or {}
    state.opp_side_conditions = opp_side_conditions or {}
    state.shadow_weather = shadow_weather or {}
    state.shadow_fields  = shadow_fields  or {}
    state.my_volatiles   = my_volatiles   or {}
    state.opp_volatiles  = opp_volatiles  or {}
    return state


# ---------------------------------------------------------------------------
# Output shape, dtype, and schema
# ---------------------------------------------------------------------------

class TestOutputShape(unittest.TestCase):
    def test_tokens_shape(self):
        tokens, field, ap = encode_state(_make_state())
        self.assertEqual(tokens.shape, (12, N_MON_FEATURES))

    def test_field_shape(self):
        _, field, _ = encode_state(_make_state())
        self.assertEqual(field.shape, (N_FIELD_FEATURES,))

    def test_active_pair_shape(self):
        _, _, ap = encode_state(_make_state())
        self.assertEqual(ap.shape, (N_ACTIVE_PAIR_FEATURES,))

    def test_flat_shape(self):
        flat = encode_state_flat(_make_state())
        self.assertEqual(flat.shape, (N_TOTAL,))

    def test_total_is_correct(self):
        self.assertEqual(N_TOTAL, 12 * N_MON_FEATURES + N_FIELD_FEATURES + N_ACTIVE_PAIR_FEATURES)
        self.assertEqual(N_TOTAL, 410)

    def test_dtypes_float32(self):
        tokens, field, ap = encode_state(_make_state())
        self.assertEqual(tokens.dtype, np.float32)
        self.assertEqual(field.dtype,  np.float32)
        self.assertEqual(ap.dtype,     np.float32)

    def test_no_nan_or_inf(self):
        tokens, field, ap = encode_state(_make_state())
        self.assertTrue(np.all(np.isfinite(tokens)))
        self.assertTrue(np.all(np.isfinite(field)))
        self.assertTrue(np.all(np.isfinite(ap)))

    def test_feature_name_counts(self):
        self.assertEqual(len(MON_FEATURE_NAMES),         N_MON_FEATURES)
        self.assertEqual(len(FIELD_FEATURE_NAMES),       N_FIELD_FEATURES)
        self.assertEqual(len(ACTIVE_PAIR_FEATURE_NAMES), N_ACTIVE_PAIR_FEATURES)


# ---------------------------------------------------------------------------
# My-team token encoding
# ---------------------------------------------------------------------------

class TestMyTeam(unittest.TestCase):
    def _idx(self, name):
        return MON_FEATURE_NAMES.index(name)

    def test_active_flag_at_slot_0(self):
        tokens, _, _ = encode_state(_make_state())
        self.assertEqual(tokens[0, self._idx("is_active")], 1.0)
        for i in range(1, 6):
            self.assertEqual(tokens[i, self._idx("is_active")], 0.0)

    def test_hp_fraction_stored(self):
        team = [_mon("a", hp_frac=0.75)] + [_mon(f"m{i}") for i in range(5)]
        tokens, _, _ = encode_state(_make_state(my_team=team))
        self.assertAlmostEqual(tokens[0, self._idx("hp_frac")], 0.75, places=4)

    def test_fainted_slot_flags(self):
        team = [_mon("alive")] + [_mon("dead", hp_frac=0.0)] + [_mon(f"m{i}") for i in range(4)]
        my_hp = {id(p): p.current_hp_fraction for p in team}
        tokens, _, _ = encode_state(_make_state(my_team=team, my_hp=my_hp))
        fainted = [i for i in range(6) if tokens[i, self._idx("is_fainted")] == 1.0]
        self.assertEqual(len(fainted), 1)
        self.assertEqual(tokens[fainted[0], self._idx("hp_frac")], 0.0)
        self.assertEqual(tokens[fainted[0], self._idx("best_dmg_frac")], 0.0)

    def test_status_burn(self):
        team = [_mon("burny")] + [_mon(f"m{i}") for i in range(5)]
        tokens, _, _ = encode_state(_make_state(my_team=team, my_status={id(team[0]): Status.BRN}))
        self.assertEqual(tokens[0, self._idx("status_burn")],   1.0)
        self.assertEqual(tokens[0, self._idx("status_poison")], 0.0)

    def test_status_paralysis(self):
        team = [_mon("paraly")] + [_mon(f"m{i}") for i in range(5)]
        tokens, _, _ = encode_state(_make_state(my_team=team, my_status={id(team[0]): Status.PAR}))
        self.assertEqual(tokens[0, self._idx("status_paralysis")], 1.0)

    def test_stat_boosts_normalized(self):
        team = [_mon("boosted")] + [_mon(f"m{i}") for i in range(5)]
        boosts = {id(team[0]): {"atk": 6, "spa": -3, "spe": 0, "def": 0, "spd": 0}}
        tokens, _, _ = encode_state(_make_state(my_team=team, my_boosts=boosts))
        self.assertAlmostEqual(tokens[0, self._idx("atk_boost")],   1.0,  places=4)
        self.assertAlmostEqual(tokens[0, self._idx("spa_boost")], -0.5,  places=4)
        self.assertAlmostEqual(tokens[0, self._idx("spe_boost")],   0.0,  places=4)

    def test_move_flag_setup(self):
        sd = _move("swordsdance", bp=0, category=MoveCategory.STATUS)
        sd.id = "swordsdance"
        team = [_mon("sweeper", moves={"swordsdance": sd})] + [_mon(f"m{i}") for i in range(5)]
        tokens, _, _ = encode_state(_make_state(my_team=team))
        self.assertEqual(tokens[0, self._idx("has_setup")], 1.0)

    def test_move_count_full(self):
        moves = {f"move{i}": _move(f"move{i}") for i in range(4)}
        team = [_mon("fullset", moves=moves)] + [_mon(f"m{i}") for i in range(5)]
        tokens, _, _ = encode_state(_make_state(my_team=team))
        self.assertAlmostEqual(tokens[0, self._idx("move_count_known")], 1.0, places=4)

    def test_move_count_partial(self):
        team = [_mon("partial", moves={"tackle": _move("tackle")})] + [_mon(f"m{i}") for i in range(5)]
        tokens, _, _ = encode_state(_make_state(my_team=team))
        self.assertAlmostEqual(tokens[0, self._idx("move_count_known")], 0.25, places=4)

    def test_no_belief_entropy_for_own_team(self):
        tokens, _, _ = encode_state(_make_state())
        for i in range(6):
            self.assertEqual(tokens[i, MON_FEATURE_NAMES.index("belief_entropy")], 0.0)


# ---------------------------------------------------------------------------
# Opponent token encoding
# ---------------------------------------------------------------------------

class TestOpponentTeam(unittest.TestCase):
    def _idx(self, name):
        return MON_FEATURE_NAMES.index(name)

    def test_unseen_slots_max_entropy(self):
        opp_mon = _mon("gengar")
        opp_slots = [opp_mon, None, None, None, None, None]
        state = _make_state(opp_team=[opp_mon], opp_slots=opp_slots,
                            opp_hp={id(opp_mon): 1.0})
        tokens, _, _ = encode_state(state)
        for i in range(7, 12):
            self.assertEqual(tokens[i, self._idx("belief_entropy")], 1.0)

    def test_unseen_slots_full_hp(self):
        opp_mon = _mon("gengar")
        opp_slots = [opp_mon, None, None, None, None, None]
        state = _make_state(opp_team=[opp_mon], opp_slots=opp_slots,
                            opp_hp={id(opp_mon): 1.0})
        tokens, _, _ = encode_state(state)
        for i in range(7, 12):
            self.assertEqual(tokens[i, self._idx("hp_frac")], 1.0)

    def test_active_opponent_at_index_6(self):
        opp_slots = [None, _mon("gengar"), None, None, None, None]
        opp_hp = {id(opp_slots[1]): 0.5}
        state = _make_state(opp_team=[opp_slots[1]], opp_slots=opp_slots,
                            opp_active_idx=1, opp_hp=opp_hp)
        tokens, _, _ = encode_state(state)
        self.assertEqual(tokens[6, self._idx("is_active")], 1.0)
        self.assertAlmostEqual(tokens[6, self._idx("hp_frac")], 0.5, places=4)

    def test_belief_entropy_uniform_4(self):
        """Uniform over 4 candidates → entropy = 1.0."""
        opp_mon = _mon("gengar")
        belief = Mock()
        belief.dist = [(Mock(), 0.25)] * 4
        opp_slots = [opp_mon, None, None, None, None, None]
        state = _make_state(opp_team=[opp_mon], opp_slots=opp_slots,
                            opp_hp={id(opp_mon): 1.0},
                            opp_beliefs={id(opp_mon): belief})
        tokens, _, _ = encode_state(state)
        self.assertAlmostEqual(tokens[6, self._idx("belief_entropy")], 1.0, places=4)

    def test_belief_entropy_certain(self):
        opp_mon = _mon("gengar")
        belief = Mock()
        belief.dist = [(Mock(), 1.0)]
        opp_slots = [opp_mon, None, None, None, None, None]
        state = _make_state(opp_team=[opp_mon], opp_slots=opp_slots,
                            opp_hp={id(opp_mon): 1.0},
                            opp_beliefs={id(opp_mon): belief})
        tokens, _, _ = encode_state(state)
        self.assertAlmostEqual(tokens[6, self._idx("belief_entropy")], 0.0, places=4)

    def test_fallback_without_opp_slots(self):
        state = _make_state(opp_slots=None)
        tokens, _, _ = encode_state(state)
        self.assertEqual(tokens.shape, (12, N_MON_FEATURES))


# ---------------------------------------------------------------------------
# Volatile condition encoding
# ---------------------------------------------------------------------------

class TestVolatileConditions(unittest.TestCase):
    def _idx(self, name):
        return MON_FEATURE_NAMES.index(name)

    def _active_team(self, active_mon):
        return [active_mon] + [_mon(f"bench{i}") for i in range(5)]

    def test_substitute_active(self):
        active = _mon("gardevoir", effects={Effect.SUBSTITUTE: 1})
        team = self._active_team(active)
        tokens, _, _ = encode_state(_make_state(my_team=team))
        self.assertEqual(tokens[0, self._idx("has_substitute")], 1.0)

    def test_substitute_not_on_bench(self):
        """Bench Pokemon should never have volatile features set."""
        bench = _mon("blissey", effects={Effect.SUBSTITUTE: 1})
        team = [_mon("active")] + [bench] + [_mon(f"p{i}") for i in range(4)]
        tokens, _, _ = encode_state(_make_state(my_team=team))
        self.assertEqual(tokens[1, self._idx("has_substitute")], 0.0)

    def test_leech_seed(self):
        active = _mon("tyranitar", effects={Effect.LEECH_SEED: 1})
        team = self._active_team(active)
        tokens, _, _ = encode_state(_make_state(my_team=team))
        self.assertEqual(tokens[0, self._idx("has_leech_seed")], 1.0)

    def test_taunt(self):
        active = _mon("garchomp", effects={Effect.TAUNT: 3})
        team = self._active_team(active)
        tokens, _, _ = encode_state(_make_state(my_team=team))
        self.assertEqual(tokens[0, self._idx("is_taunted")], 1.0)

    def test_encore(self):
        active = _mon("lopunny", effects={Effect.ENCORE: 2})
        team = self._active_team(active)
        tokens, _, _ = encode_state(_make_state(my_team=team))
        self.assertEqual(tokens[0, self._idx("is_encored")], 1.0)

    def test_confusion_from_effects(self):
        active = _mon("infernape", effects={Effect.CONFUSION: 1})
        team = self._active_team(active)
        tokens, _, _ = encode_state(_make_state(my_team=team))
        self.assertEqual(tokens[0, self._idx("is_confused")], 1.0)

    def test_confusion_from_shadow_volatiles(self):
        """confusion_turns in ShadowState should also set is_confused."""
        active = _mon("infernape")
        team = self._active_team(active)
        my_vol = {id(active): {"confusion_turns": 3}}
        tokens, _, _ = encode_state(_make_state(my_team=team, my_volatiles=my_vol))
        self.assertEqual(tokens[0, self._idx("is_confused")], 1.0)

    def test_no_confusion_zero_turns(self):
        active = _mon("infernape")
        team = self._active_team(active)
        my_vol = {id(active): {"confusion_turns": 0}}
        tokens, _, _ = encode_state(_make_state(my_team=team, my_volatiles=my_vol))
        self.assertEqual(tokens[0, self._idx("is_confused")], 0.0)

    def test_perish_song_3_turns(self):
        active = _mon("gengar", effects={Effect.PERISH3: 1})
        team = self._active_team(active)
        tokens, _, _ = encode_state(_make_state(my_team=team))
        self.assertAlmostEqual(tokens[0, self._idx("perish_turns")], 1.0, places=4)

    def test_perish_song_1_turn(self):
        active = _mon("gengar", effects={Effect.PERISH1: 1})
        team = self._active_team(active)
        tokens, _, _ = encode_state(_make_state(my_team=team))
        self.assertAlmostEqual(tokens[0, self._idx("perish_turns")], 1/3, delta=0.01)

    def test_perish_song_0(self):
        active = _mon("gengar", effects={Effect.PERISH0: 1})
        team = self._active_team(active)
        tokens, _, _ = encode_state(_make_state(my_team=team))
        self.assertAlmostEqual(tokens[0, self._idx("perish_turns")], 0.0, places=4)

    def test_no_perish_default(self):
        active = _mon("gengar")
        team = self._active_team(active)
        tokens, _, _ = encode_state(_make_state(my_team=team))
        self.assertEqual(tokens[0, self._idx("perish_turns")], 0.0)

    def test_yawn(self):
        active = _mon("slowpoke", effects={Effect.YAWN: 1})
        team = self._active_team(active)
        tokens, _, _ = encode_state(_make_state(my_team=team))
        self.assertEqual(tokens[0, self._idx("yawn_incoming")], 1.0)

    def test_terastallized(self):
        active = _mon("dragapult", is_terastallized=True)
        team = self._active_team(active)
        tokens, _, _ = encode_state(_make_state(my_team=team))
        self.assertEqual(tokens[0, self._idx("is_terastallized")], 1.0)

    def test_not_terastallized_by_default(self):
        tokens, _, _ = encode_state(_make_state())
        self.assertEqual(tokens[0, self._idx("is_terastallized")], 0.0)

    def test_no_volatiles_on_default_mon(self):
        """All volatile features should be 0 on a plain mon with no effects."""
        tokens, _, _ = encode_state(_make_state())
        vol_features = ["has_substitute", "has_leech_seed", "is_taunted",
                        "is_encored", "is_confused", "perish_turns",
                        "yawn_incoming", "is_terastallized"]
        for fname in vol_features:
            self.assertEqual(tokens[0, self._idx(fname)], 0.0, msg=f"{fname} should be 0")

    def test_opponent_active_volatile(self):
        """Volatiles should also be encoded for opponent active slot."""
        opp_active = _mon("lucario", effects={Effect.TAUNT: 2})
        opp_slots = [opp_active, None, None, None, None, None]
        state = _make_state(opp_team=[opp_active], opp_slots=opp_slots,
                            opp_hp={id(opp_active): 1.0})
        tokens, _, _ = encode_state(state)
        self.assertEqual(tokens[6, self._idx("is_taunted")], 1.0)

    def test_multiple_volatiles_simultaneously(self):
        active = _mon("togekiss", effects={
            Effect.SUBSTITUTE: 1,
            Effect.ENCORE: 2,
            Effect.YAWN: 1,
        }, is_terastallized=True)
        team = self._active_team(active)
        tokens, _, _ = encode_state(_make_state(my_team=team))
        self.assertEqual(tokens[0, self._idx("has_substitute")],  1.0)
        self.assertEqual(tokens[0, self._idx("is_encored")],      1.0)
        self.assertEqual(tokens[0, self._idx("yawn_incoming")],   1.0)
        self.assertEqual(tokens[0, self._idx("is_terastallized")], 1.0)


# ---------------------------------------------------------------------------
# Tera field features
# ---------------------------------------------------------------------------

class TestTeraFieldFeatures(unittest.TestCase):
    def _fidx(self, name):
        return FIELD_FEATURE_NAMES.index(name)

    def test_my_tera_used_false_by_default(self):
        _, field, _ = encode_state(_make_state())
        self.assertEqual(field[self._fidx("my_tera_used")], 0.0)

    def test_opp_tera_used_false_by_default(self):
        _, field, _ = encode_state(_make_state())
        self.assertEqual(field[self._fidx("opp_tera_used")], 0.0)

    def test_my_tera_used_true(self):
        _, field, _ = encode_state(_make_state(used_tera=True))
        self.assertEqual(field[self._fidx("my_tera_used")], 1.0)

    def test_opp_tera_used_true(self):
        _, field, _ = encode_state(_make_state(opponent_used_tera=True))
        self.assertEqual(field[self._fidx("opp_tera_used")], 1.0)

    def test_both_tera_independent(self):
        _, field, _ = encode_state(_make_state(used_tera=True, opponent_used_tera=False))
        self.assertEqual(field[self._fidx("my_tera_used")],  1.0)
        self.assertEqual(field[self._fidx("opp_tera_used")], 0.0)


# ---------------------------------------------------------------------------
# Field features
# ---------------------------------------------------------------------------

class TestFieldFeatures(unittest.TestCase):
    def _fidx(self, name):
        return FIELD_FEATURE_NAMES.index(name)

    def _wkey(self, name):
        k = Mock(); k.name = name; return k

    def _fkey(self, name):
        k = Mock(); k.name = name; return k

    def test_sun_weather(self):
        _, field, _ = encode_state(_make_state(shadow_weather={self._wkey("sunnyday"): 5}))
        self.assertEqual(field[self._fidx("weather_sun")],  1.0)
        self.assertEqual(field[self._fidx("weather_rain")], 0.0)

    def test_rain_weather(self):
        _, field, _ = encode_state(_make_state(shadow_weather={self._wkey("raindance"): 3}))
        self.assertEqual(field[self._fidx("weather_rain")], 1.0)

    def test_sand_weather(self):
        _, field, _ = encode_state(_make_state(shadow_weather={self._wkey("sandstorm"): 5}))
        self.assertEqual(field[self._fidx("weather_sand")], 1.0)

    def test_no_weather_at_zero_turns(self):
        _, field, _ = encode_state(_make_state(shadow_weather={self._wkey("sunnyday"): 0}))
        self.assertEqual(field[self._fidx("weather_sun")], 0.0)

    def test_electric_terrain(self):
        _, field, _ = encode_state(_make_state(shadow_fields={self._fkey("electricterrain"): 5}))
        self.assertEqual(field[self._fidx("terrain_electric")], 1.0)

    def test_my_stealth_rock(self):
        sc = Mock(); sc.name = "STEALTH_ROCK"
        _, field, _ = encode_state(_make_state(my_side_conditions={sc: 1}))
        self.assertEqual(field[self._fidx("my_stealth_rock")],  1.0)
        self.assertEqual(field[self._fidx("opp_stealth_rock")], 0.0)

    def test_opp_stealth_rock(self):
        sc = Mock(); sc.name = "STEALTH_ROCK"
        _, field, _ = encode_state(_make_state(opp_side_conditions={sc: 1}))
        self.assertEqual(field[self._fidx("opp_stealth_rock")], 1.0)
        self.assertEqual(field[self._fidx("my_stealth_rock")],  0.0)

    def test_spikes_normalized(self):
        sc = Mock(); sc.name = "spikes"
        _, field3, _ = encode_state(_make_state(my_side_conditions={sc: 3}))
        self.assertAlmostEqual(field3[self._fidx("my_spikes")], 1.0, places=4)
        _, field1, _ = encode_state(_make_state(my_side_conditions={sc: 1}))
        self.assertAlmostEqual(field1[self._fidx("my_spikes")], 1/3, delta=0.01)

    def test_toxic_spikes_normalized(self):
        sc = Mock(); sc.name = "toxicspikes"
        _, field, _ = encode_state(_make_state(my_side_conditions={sc: 2}))
        self.assertAlmostEqual(field[self._fidx("my_toxic_spikes")], 1.0, places=4)

    def test_no_weather_terrain_hazards_by_default(self):
        _, field, _ = encode_state(_make_state())
        np.testing.assert_array_equal(field[:16], 0.0)


# ---------------------------------------------------------------------------
# Team-level summary signals
# ---------------------------------------------------------------------------

class TestTeamSummarySignals(unittest.TestCase):
    def _fidx(self, name):
        return FIELD_FEATURE_NAMES.index(name)

    def test_moves_first_faster(self):
        fast = _mon("fast", spe=150)
        slow = _mon("slow", spe=50)
        my_team   = [fast] + [_mon(f"p{i}") for i in range(5)]
        opp_slots = [slow, None, None, None, None, None]
        state = _make_state(my_team=my_team, opp_team=[slow], opp_slots=opp_slots,
                            opp_hp={id(slow): 1.0})
        _, field, _ = encode_state(state)
        self.assertEqual(field[self._fidx("moves_first")], 1.0)

    def test_moves_first_slower(self):
        slow = _mon("slow", spe=50)
        fast = _mon("fast", spe=150)
        my_team   = [slow] + [_mon(f"p{i}") for i in range(5)]
        opp_slots = [fast, None, None, None, None, None]
        state = _make_state(my_team=my_team, opp_team=[fast], opp_slots=opp_slots,
                            opp_hp={id(fast): 1.0})
        _, field, _ = encode_state(state)
        self.assertEqual(field[self._fidx("moves_first")], -1.0)

    def test_moves_first_tie(self):
        a = _mon("a"); b = _mon("b")
        my_team   = [a] + [_mon(f"p{i}") for i in range(5)]
        opp_slots = [b, None, None, None, None, None]
        state = _make_state(my_team=my_team, opp_team=[b], opp_slots=opp_slots,
                            opp_hp={id(b): 1.0})
        _, field, _ = encode_state(state)
        self.assertEqual(field[self._fidx("moves_first")], 0.0)

    def test_team_hp_diff_equal(self):
        team = [_mon(f"p{i}", hp_frac=1.0) for i in range(6)]
        opp  = [_mon(f"o{i}", hp_frac=1.0) for i in range(6)]
        state = _make_state(my_team=team, opp_team=opp, opp_slots=list(opp),
                            my_hp={id(p): 1.0 for p in team},
                            opp_hp={id(p): 1.0 for p in opp})
        _, field, _ = encode_state(state)
        self.assertAlmostEqual(field[self._fidx("team_hp_diff")], 0.0, delta=0.01)

    def test_team_hp_diff_advantage(self):
        my_team = [_mon(f"p{i}", hp_frac=1.0) for i in range(6)]
        opp     = [_mon(f"o{i}", hp_frac=0.5) for i in range(6)]
        state = _make_state(my_team=my_team, opp_team=opp, opp_slots=list(opp),
                            my_hp={id(p): 1.0 for p in my_team},
                            opp_hp={id(p): 0.5 for p in opp})
        _, field, _ = encode_state(state)
        self.assertGreater(field[self._fidx("team_hp_diff")], 0.0)

    def test_alive_count_diff_symmetric(self):
        team = [_mon(f"p{i}") for i in range(6)]
        opp  = [_mon(f"o{i}") for i in range(6)]
        state = _make_state(my_team=team, opp_team=opp, opp_slots=list(opp),
                            my_hp={id(p): 1.0 for p in team},
                            opp_hp={id(p): 1.0 for p in opp})
        _, field, _ = encode_state(state)
        self.assertAlmostEqual(field[self._fidx("alive_count_diff")], 0.0, delta=0.01)

    def test_best_switch_safety_in_range(self):
        _, field, _ = encode_state(_make_state())
        self.assertGreaterEqual(field[self._fidx("best_switch_safety")], 0.0)
        self.assertLessEqual(field[self._fidx("best_switch_safety")],    1.0)

    def test_hazard_pressure_no_hazards(self):
        _, field, _ = encode_state(_make_state())
        self.assertEqual(field[self._fidx("hazard_pressure")], 0.0)

    def test_hazard_pressure_opp_rocks(self):
        sc = Mock(); sc.name = "stealthrock"
        _, field, _ = encode_state(_make_state(opp_side_conditions={sc: 1}))
        self.assertGreater(field[self._fidx("hazard_pressure")], 0.0)

    def test_hazard_pressure_my_rocks(self):
        sc = Mock(); sc.name = "stealthrock"
        _, field, _ = encode_state(_make_state(my_side_conditions={sc: 1}))
        self.assertLess(field[self._fidx("hazard_pressure")], 0.0)

    def test_hazard_pressure_in_range(self):
        sc = Mock(); sc.name = "stealthrock"
        _, field, _ = encode_state(_make_state(opp_side_conditions={sc: 1}))
        self.assertGreaterEqual(field[self._fidx("hazard_pressure")], -1.0)
        self.assertLessEqual(field[self._fidx("hazard_pressure")],     1.0)


# ---------------------------------------------------------------------------
# Active pair features
# ---------------------------------------------------------------------------

class TestActivePair(unittest.TestCase):
    def _apidx(self, name):
        return ACTIVE_PAIR_FEATURE_NAMES.index(name)

    def test_all_binary(self):
        _, _, ap = encode_state(_make_state())
        for v in ap:
            self.assertIn(v, (0.0, 1.0))

    def test_speed_tie_detected(self):
        a = _mon("a"); b = _mon("b")  # both spe=80
        my_team   = [a] + [_mon(f"p{i}") for i in range(5)]
        opp_slots = [b, None, None, None, None, None]
        state = _make_state(my_team=my_team, opp_team=[b], opp_slots=opp_slots,
                            opp_hp={id(b): 1.0})
        _, _, ap = encode_state(state)
        self.assertEqual(ap[self._apidx("speed_tie_or_unknown")], 1.0)

    def test_speed_tie_absent_when_different(self):
        fast = _mon("fast", spe=150)
        slow = _mon("slow", spe=50)
        my_team   = [fast] + [_mon(f"p{i}") for i in range(5)]
        opp_slots = [slow, None, None, None, None, None]
        state = _make_state(my_team=my_team, opp_team=[slow], opp_slots=opp_slots,
                            opp_hp={id(slow): 1.0})
        _, _, ap = encode_state(state)
        self.assertEqual(ap[self._apidx("speed_tie_or_unknown")], 0.0)

    def test_can_ko_zero_without_dmg_estimator(self):
        _, _, ap = encode_state(_make_state())
        self.assertEqual(ap[self._apidx("can_ko")], 0.0)

    def test_can_be_ko_zero_without_dmg_estimator(self):
        _, _, ap = encode_state(_make_state())
        self.assertEqual(ap[self._apidx("can_be_ko")], 0.0)


# ---------------------------------------------------------------------------
# Flat encoding consistency
# ---------------------------------------------------------------------------

class TestFlatEncoding(unittest.TestCase):
    def test_flat_matches_concat(self):
        state = _make_state()
        tokens, field, ap = encode_state(state)
        flat = encode_state_flat(state)
        expected = np.concatenate([tokens.reshape(-1), field, ap])
        np.testing.assert_array_almost_equal(flat, expected)

    def test_flat_no_nan(self):
        self.assertTrue(np.all(np.isfinite(encode_state_flat(_make_state()))))

    def test_flat_length(self):
        self.assertEqual(encode_state_flat(_make_state()).shape, (410,))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):
    def test_empty_teams(self):
        state = _make_state(my_team=[], opp_team=[])
        state.my_active = state.opp_active = None
        state.my_hp = state.opp_hp = {}
        state.opp_slots = []
        tokens, field, ap = encode_state(state)
        self.assertEqual(tokens.shape, (12, N_MON_FEATURES))
        self.assertEqual(field.shape,  (N_FIELD_FEATURES,))
        self.assertEqual(ap.shape,     (N_ACTIVE_PAIR_FEATURES,))

    def test_all_fainted(self):
        team = [_mon(f"p{i}", hp_frac=0.0) for i in range(6)]
        my_hp = {id(p): 0.0 for p in team}
        tokens, _, _ = encode_state(_make_state(my_team=team, my_hp=my_hp))
        for i in range(6):
            self.assertEqual(tokens[i, MON_FEATURE_NAMES.index("is_fainted")], 1.0)

    def test_values_in_range(self):
        tokens, field, ap = encode_state(_make_state())

        # Token boost cols (9-13) are [-1,1]; everything else is [0,1].
        boost_cols = list(range(9, 14))
        other_cols = [i for i in range(N_MON_FEATURES) if i not in boost_cols]
        self.assertTrue(np.all(tokens[:, other_cols] >= 0.0))
        self.assertTrue(np.all(tokens[:, other_cols] <= 1.0))
        self.assertTrue(np.all(tokens[:, boost_cols] >= -1.0))
        self.assertTrue(np.all(tokens[:, boost_cols] <= 1.0))

        # Field: indices 0-15 are [0,1]; indices 16+ are [-1,1].
        self.assertTrue(np.all(field[:16] >= 0.0))
        self.assertTrue(np.all(field[:16] <= 1.0))
        self.assertTrue(np.all(field[16:] >= -1.0))
        self.assertTrue(np.all(field[16:] <= 1.0))

        # Active pair: all {0,1}.
        for v in ap:
            self.assertIn(v, (0.0, 1.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
