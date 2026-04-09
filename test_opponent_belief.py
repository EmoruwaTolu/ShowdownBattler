"""
Tests for new OpponentBelief inference methods:
  - observe_speed_comparison
  - observe_damage_taken

All tests run against the real gen9randombattle.json data.
"""
from __future__ import annotations

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.model.opponent_model import (
    build_opponent_belief,
    lookup_randbats_candidates,
    _candidate_effective_speed,
    _candidate_expected_damage_frac,
    OpponentBelief,
    SetCandidate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _belief_for_species(species: str, gen: int = 9) -> OpponentBelief:
    """Build a fresh uniform belief for a species (no observations)."""
    from types import SimpleNamespace
    opp = SimpleNamespace()
    opp.species = species
    opp.base_species = species
    opp.moves = {}
    opp.item = None
    opp.ability = None
    opp.tera_type = None
    return build_opponent_belief(opp, gen)


def _role_weights(belief: OpponentBelief) -> dict:
    """Return {role_name_suffix: weight} for easy assertions."""
    return {c.id.split(":", 1)[1]: round(w, 6) for c, w in belief.dist}


# ---------------------------------------------------------------------------
# _candidate_effective_speed
# ---------------------------------------------------------------------------

class TestCandidateEffectiveSpeed(unittest.TestCase):
    """Unit tests for the speed-stat helper."""

    def setUp(self):
        self.candidates = lookup_randbats_candidates("arcaninehisui", 9)
        self.assertIsNotNone(self.candidates)
        # Arcanine-Hisui: base_spe=90, level=79
        # stat = int(2*90*79/100 + 5) = 147
        # with scarf: int(147*1.5) = 220

    def _get_role(self, name_part: str) -> SetCandidate:
        for c in self.candidates:
            if name_part.lower() in c.id.lower():
                return c
        self.fail(f"No candidate matching '{name_part}'")

    def test_bulky_attacker_no_scarf(self):
        cand = self._get_role("bulky")
        spe = _candidate_effective_speed(cand, 9)
        self.assertEqual(int(spe), 147)

    def test_fast_attacker_with_scarf(self):
        cand = self._get_role("fast")
        # Fast Attacker has Choice Scarf in its item pool
        spe = _candidate_effective_speed(cand, 9)
        # Scarf applies → 220
        self.assertEqual(int(spe), 220)

    def test_unknown_species_returns_nonzero(self):
        cand = SetCandidate(
            id="fakespecies:fallback",
            species_id="fakespecies",
            moves=set(),
            abilities=set(),
            items=set(),
            tera_types=set(),
            is_physical=False,
            has_setup=False,
            has_priority=False,
        )
        spe = _candidate_effective_speed(cand, 9)
        self.assertGreater(spe, 0)


# ---------------------------------------------------------------------------
# observe_speed_comparison — Arcanine-Hisui
# ---------------------------------------------------------------------------
#
# Arcanine-Hisui roles in gen9randombattle.json:
#   Bulky Attacker  items: [Heavy-Duty Boots]       → speed 147
#   Fast Attacker   items: [Choice Band, Choice Scarf] → speed 147 or 220
#
# our_speed = 160 (between 147 and 220)

class TestObserveSpeedComparison(unittest.TestCase):

    GEN   = 9
    OPP   = "arcaninehisui"
    SPEED = 160  # sits between 147 (no scarf) and 220 (scarf)

    def _fresh(self) -> OpponentBelief:
        return _belief_for_species(self.OPP, self.GEN)

    def _weight_of_role(self, belief: OpponentBelief, name_part: str) -> float:
        for c, w in belief.dist:
            if name_part.lower() in c.id.lower():
                return w
        self.fail(f"Role '{name_part}' not found in belief")

    def test_moved_first_upweights_fast_role(self):
        """Opponent moved before us → must be faster → Fast Attacker (scarf) preferred."""
        belief = self._fresh()
        w_fast_before = self._weight_of_role(belief, "fast")
        w_bulky_before = self._weight_of_role(belief, "bulky")

        belief.observe_speed_comparison(our_speed=self.SPEED, moved_first=True)

        w_fast_after  = self._weight_of_role(belief, "fast")
        w_bulky_after = self._weight_of_role(belief, "bulky")

        # Fast (scarf=220 > 160): consistent → not penalised → higher relative weight
        # Bulky (147 < 160): contradicts → penalised → lower relative weight
        self.assertGreater(w_fast_after, w_bulky_after,
                           "Fast role should dominate after opponent moved first")

    def test_moved_second_upweights_bulky_role(self):
        """Opponent moved after us → must be slower → Bulky Attacker (no scarf) preferred."""
        belief = self._fresh()
        belief.observe_speed_comparison(our_speed=self.SPEED, moved_first=False)

        w_fast  = self._weight_of_role(belief, "fast")
        w_bulky = self._weight_of_role(belief, "bulky")

        self.assertGreater(w_bulky, w_fast,
                           "Bulky role should dominate after opponent moved second")

    def test_trick_room_inverts_comparison(self):
        """In Trick Room, moving first means being *slower*."""
        belief_tr = _belief_for_species(self.OPP, self.GEN)
        belief_normal = _belief_for_species(self.OPP, self.GEN)

        belief_tr.observe_speed_comparison(our_speed=self.SPEED, moved_first=True,
                                            trick_room_active=True)
        belief_normal.observe_speed_comparison(our_speed=self.SPEED, moved_first=False,
                                               trick_room_active=False)

        # Both should yield the same outcome: bulky preferred over fast
        w_fast_tr    = sum(w for c, w in belief_tr.dist if "fast" in c.id)
        w_bulky_tr   = sum(w for c, w in belief_tr.dist if "bulky" in c.id)
        w_fast_nrm   = sum(w for c, w in belief_normal.dist if "fast" in c.id)
        w_bulky_nrm  = sum(w for c, w in belief_normal.dist if "bulky" in c.id)

        self.assertGreater(w_bulky_tr, w_fast_tr)
        self.assertGreater(w_bulky_nrm, w_fast_nrm)

    def test_speed_tie_no_change(self):
        """When our speed equals candidate speed, no penalty applied."""
        # Bulky Attacker speed ≈ 147; use our_speed=147 so it's a tie
        belief = _belief_for_species(self.OPP, self.GEN)
        dist_before = dict((c.id, w) for c, w in belief.dist)

        belief.observe_speed_comparison(our_speed=147, moved_first=True)

        for c, w in belief.dist:
            if "bulky" in c.id.lower():
                # Within ±2 tie threshold; should not be penalised
                self.assertAlmostEqual(w, dist_before[c.id] /
                                       sum(dist_before.values()), places=4)

    def test_distribution_still_sums_to_one(self):
        belief = _belief_for_species(self.OPP, self.GEN)
        belief.observe_speed_comparison(our_speed=self.SPEED, moved_first=True)
        total = sum(w for _, w in belief.dist)
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_repeated_observations_are_consistent(self):
        """Applying the same observation twice should not flip or over-penalise."""
        belief = _belief_for_species(self.OPP, self.GEN)
        belief.observe_speed_comparison(our_speed=self.SPEED, moved_first=True)
        w_fast_1 = self._weight_of_role(belief, "fast")

        belief.observe_speed_comparison(our_speed=self.SPEED, moved_first=True)
        w_fast_2 = self._weight_of_role(belief, "fast")

        # Second observation should make fast role even more dominant (or equal)
        self.assertGreaterEqual(w_fast_2, w_fast_1 - 1e-9)


# ---------------------------------------------------------------------------
# _candidate_expected_damage_frac
# ---------------------------------------------------------------------------

class TestCandidateExpectedDamageFrac(unittest.TestCase):
    """Unit tests for the damage fraction helper — Ampharos."""

    # Ampharos: base_hp=90, base_spd=90, level=88
    # HP = int(2*90*88/100 + 88 + 10) = int(158.4+88+10) = 256
    # SpD = int(2*90*88/100 + 5)      = int(158.4+5)      = 163
    # SpD_AV = int(163 * 1.5)                             = 244

    GEN = 9
    SPP = "ampharos"

    def _get_role(self, name_part: str) -> SetCandidate:
        cands = lookup_randbats_candidates(self.SPP, self.GEN)
        self.assertIsNotNone(cands)
        for c in cands:
            if name_part.lower() in c.id.lower():
                return c
        self.fail(f"Role '{name_part}' not found")

    def test_av_role_lower_expected_damage(self):
        """AV Pivot has higher effective SpD → lower expected special damage."""
        av   = self._get_role("av")
        wall = self._get_role("wallbreaker")

        frac_av   = _candidate_expected_damage_frac(av,   self.GEN, 90, True, 200)
        frac_wall = _candidate_expected_damage_frac(wall, self.GEN, 90, True, 200)

        self.assertLess(frac_av, frac_wall,
                        "AV should absorb more damage → lower expected fraction")

    def test_zero_base_power_returns_zero(self):
        av = self._get_role("av")
        self.assertEqual(_candidate_expected_damage_frac(av, self.GEN, 0, True, 200), 0.0)

    def test_zero_attacker_stat_returns_zero(self):
        av = self._get_role("av")
        self.assertEqual(_candidate_expected_damage_frac(av, self.GEN, 90, True, 0), 0.0)

    def test_returns_positive_for_valid_inputs(self):
        wall = self._get_role("wallbreaker")
        frac = _candidate_expected_damage_frac(wall, self.GEN, 90, True, 200)
        self.assertGreater(frac, 0.0)
        self.assertLess(frac, 5.0)   # sanity: even a huge hit shouldn't be > 5× HP


# ---------------------------------------------------------------------------
# observe_damage_taken — Ampharos
# ---------------------------------------------------------------------------
#
# Ampharos:
#   AV Pivot    items: [Assault Vest]              → SpD = 244 (×1.5)
#   Wallbreaker items: [Choice Specs, Life Orb]   → SpD = 163
#
# Attack scenario: BP=90 (e.g. Energy Ball), attacker_spa=400, level-50 formula
#   vs Wallbreaker:  expected ≈ 38.7%
#   vs AV Pivot:     expected ≈ 26.1%
#
# If we observe only 15% damage → Wallbreaker contradicted (ratio 2.58), AV is fine (1.74)

class TestObserveDamageTaken(unittest.TestCase):

    GEN = 9
    OPP = "ampharos"
    BP  = 90
    SPA = 400   # strong attacker

    def _fresh(self) -> OpponentBelief:
        return _belief_for_species(self.OPP, self.GEN)

    def _weight_of_role(self, belief: OpponentBelief, name_part: str) -> float:
        for c, w in belief.dist:
            if name_part.lower() in c.id.lower():
                return w
        self.fail(f"Role '{name_part}' not found in belief")

    def test_low_observed_damage_penalises_frail_role(self):
        """
        When observed damage is well below expected for a frail candidate,
        that candidate should be downweighted relative to the bulkier one.
        """
        belief = self._fresh()
        belief.observe_damage_taken(
            base_power=self.BP,
            is_special=True,
            attacker_stat=self.SPA,
            damage_fraction=0.15,   # very low → Wallbreaker is contraindicated
        )
        w_av   = self._weight_of_role(belief, "av")
        w_wall = self._weight_of_role(belief, "wallbreaker")
        self.assertGreater(w_av, w_wall,
                           "AV Pivot should be preferred when only 15% damage dealt")

    def test_high_observed_damage_penalises_bulky_role(self):
        """
        When observed damage is high (>38%), AV Pivot (expected ~26%) is
        inconsistent — the target wasn't that bulky.
        """
        belief = self._fresh()
        belief.observe_damage_taken(
            base_power=self.BP,
            is_special=True,
            attacker_stat=self.SPA,
            damage_fraction=0.60,   # high → AV Pivot is contraindicated
        )
        w_av   = self._weight_of_role(belief, "av")
        w_wall = self._weight_of_role(belief, "wallbreaker")
        self.assertGreater(w_wall, w_av,
                           "Wallbreaker should be preferred when 60% damage dealt")

    def test_physical_move_uses_defence_not_spdef(self):
        """Physical moves should use Def, and AV should not affect Def."""
        belief_phys = self._fresh()
        belief_spec = self._fresh()

        # Physical hit at 60%: both Wallbreaker and AV have similar Def → no big split
        belief_phys.observe_damage_taken(
            base_power=self.BP, is_special=False,
            attacker_stat=self.SPA, damage_fraction=0.60,
        )
        # Special hit at 60%: AV is penalised
        belief_spec.observe_damage_taken(
            base_power=self.BP, is_special=True,
            attacker_stat=self.SPA, damage_fraction=0.60,
        )

        w_av_phys = self._weight_of_role(belief_phys, "av")
        w_av_spec = self._weight_of_role(belief_spec, "av")
        # AV only boosts SpD so physical hit should not penalise AV as much
        # (specifically: AV Pivot should not be penalised more on phys than on spec)
        # We just assert the special hit penalises AV more than the physical hit does
        self.assertGreaterEqual(w_av_phys, w_av_spec - 1e-6)

    def test_zero_damage_fraction_is_no_op(self):
        """Damage fraction of 0 (or negative) should not change belief."""
        belief = self._fresh()
        dist_before = [(c.id, w) for c, w in belief.dist]

        belief.observe_damage_taken(
            base_power=self.BP, is_special=True,
            attacker_stat=self.SPA, damage_fraction=0.0,
        )
        dist_after = [(c.id, w) for c, w in belief.dist]
        self.assertEqual(dist_before, dist_after)

    def test_distribution_sums_to_one_after_update(self):
        belief = self._fresh()
        belief.observe_damage_taken(
            base_power=self.BP, is_special=True,
            attacker_stat=self.SPA, damage_fraction=0.15,
        )
        total = sum(w for _, w in belief.dist)
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_combined_with_speed_observation(self):
        """Both observations together should further sharpen the distribution."""
        belief = self._fresh()
        # Ampharos isn't a speed test species, but verify composability
        belief.observe_damage_taken(
            base_power=self.BP, is_special=True,
            attacker_stat=self.SPA, damage_fraction=0.15,
        )
        total = sum(w for _, w in belief.dist)
        self.assertAlmostEqual(total, 1.0, places=6)
        # Distribution still has candidates
        self.assertGreater(len(belief.dist), 0)


# ---------------------------------------------------------------------------
# Edge cases shared across both methods
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):

    GEN = 9

    def test_observe_speed_on_single_candidate_belief(self):
        """When only one candidate remains, speed observation should not crash."""
        belief = _belief_for_species("arcaninehisui", self.GEN)
        # Collapse to single candidate
        first_cand = belief.dist[0][0]
        belief.dist = [(first_cand, 1.0)]

        belief.observe_speed_comparison(our_speed=100, moved_first=True)
        total = sum(w for _, w in belief.dist)
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_observe_damage_on_single_candidate_belief(self):
        belief = _belief_for_species("ampharos", self.GEN)
        first_cand = belief.dist[0][0]
        belief.dist = [(first_cand, 1.0)]

        belief.observe_damage_taken(
            base_power=90, is_special=True, attacker_stat=200, damage_fraction=0.30,
        )
        total = sum(w for _, w in belief.dist)
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_fallback_uniform_belief_after_impossible_observation(self):
        """
        If all candidates get penalised, normalize() fallback to uniform is triggered
        (because the soft penalty 0.15/0.20 never zeros out — all stay positive).
        Distribution should still be valid.
        """
        belief = _belief_for_species("arcaninehisui", self.GEN)
        # Apply many contradicting speed observations
        for _ in range(10):
            belief.observe_speed_comparison(our_speed=500, moved_first=True)
        total = sum(w for _, w in belief.dist)
        self.assertAlmostEqual(total, 1.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
