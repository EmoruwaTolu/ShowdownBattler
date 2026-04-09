# Model Improvements Plan

## 1. Richer State Encoding [ ]

**Goal:** Replace the current flat 24-scalar `eval_terms` summary with a structured,
per-Pokemon state representation that captures the full game state.

**Why the current one is limiting:**
- 24 numbers collapse all 12 Pokemon into aggregate scores (HP ratio, alive count, etc.)
- No per-slot information — the model can't distinguish "one Pokemon at 10% HP" from
  "all Pokemon at 10% HP average"
- No per-move detail per Pokemon
- Field state (weather, terrain, screens) partially encoded but shallow
- Belief uncertainty collapsed to a single entropy scalar

**New representation — 12-token encoding (6 per side):**

Each Pokemon slot gets a fixed-size feature vector (~30 features):
- `hp_frac` — current HP as fraction of max
- `is_active`, `is_fainted` — slot state flags
- `status_*` — 6 binary flags (burn, poison, toxic, paralysis, sleep, freeze)
- `stat_boost_sum` — sum of offensive stat boosts (atk/spa)
- `speed_boost` — spe stage, normalized
- `type_adv_vs_active_opp` — best type effectiveness this mon has vs the current active opponent
- `type_disadv_vs_active_opp` — worst type effectiveness the active opponent has vs this mon
- `best_dmg_frac` — estimated damage fraction of best available move
- `has_setup`, `has_recovery`, `has_priority`, `has_pivot`, `has_hazard_clear` — move role flags
- `move_count_known` — fraction of moveset revealed (0–1)
- `belief_entropy` — uncertainty about this slot (0 for own team, 0–1 for opponent)
- `expected_speed_tier` — normalized speed stat (known or belief-averaged)

Opponent unrevealed slots use belief-averaged feature expectations.

**Deliverables:**
- [x] `bot/learning/state_encoder.py` — `encode_state(state) -> (tokens, field, active_pair)` of shape `(12, 24)` + `(21,)` + `(3,)`
- [x] Unit tests covering: fainted slots, unrevealed slots, active Pokemon, field effects, KO indicators, speed tie, hazard pressure (56 passing)
- [x] Flat version `encode_state_flat(state)` that concatenates all three (312-dim, for MLP baseline)

**Final spec:** `(12, 24)` tokens + `(21,)` field + `(3,)` active pair = **312 total**
- Per-token (24): hp, active/fainted, 6 status, 5 boosts, type adv/disadv, best_dmg_frac, 5 move roles, move_count_known, belief_entropy
- Field (21): 4 weather + 4 terrain + 8 side conditions + moves_first + team_hp_diff + alive_count_diff + best_switch_safety + hazard_pressure
- Active pair (3): can_ko + can_be_ko + speed_tie_or_unknown

---

## 2. Transformer Value Model [x]

**Goal:** Replace the flat MLP value model with a Transformer that attends over the
12 Pokemon tokens produced by the new state encoder.

**Architecture:**
- Token projection (24 → d_model=64) + side embedding (my/opp team)
- 2× TransformerEncoderLayer (MHA + residual + LayerNorm + FFN + residual + LayerNorm)
- Mean-pool over 12 tokens → concat [field(21) + active_pair(3)] → output MLP → sigmoid
- Pure numpy, full analytical backprop, Adam optimizer, ~71K parameters

**Deliverables:**
- [x] `bot/learning/transformer_value.py` — model with forward, backward, Adam, save/load
- [x] `bot/learning/train_transformer.py` — training script reading `state_flat` from records
- [x] `bot/learning/collect.py` — updated to log `state_flat` (312-dim) when `shadow_state` provided
- [x] Wire `shadow_state` into `player.py`'s `record_turn` call — passes `stats["root_state"]` (the `ShadowState` built at decision time) so `state_flat` is now logged every turn
- [x] Wire `observe_damage_taken` into battle loop — `_apply_belief_observations()` in `player.py` computes the opponent's HP delta from the previous turn and calls it whenever we used a damaging move and the same opponent is still active
- [x] Wire `observe_speed_comparison` into battle loop — `_who_moved_first()` parses `battle.observations[turn-1].events` for `|move|` messages to determine real turn order, then calls `observe_speed_comparison` with our computed effective speed and Trick Room flag
- [ ] Benchmark transformer vs MLP once new data is collected

**Interface:**
- `model.predict_value_from_state(state)` — drop-in for MCTS (same [-1,1] output as ValueModel)
- `model.save(path)` / `TransformerValueModel.load(path)` — .npz persistence

---

## 3. Tighter Belief Updating [x]

**Goal:** Make `OpponentBelief` constrain more aggressively on indirect evidence.

**Missing inferences currently:**
- **Speed tier:** if opponent outspeeds us, eliminate all sets slower than our current speed
- **Stat range:** if opponent survives a hit we know should 2HKO, they must have >X HP EVs
- **Move legality:** revealed moves narrow not just the set but the ability/item pool
  (e.g., if they used Knock Off, they probably don't have a Z-crystal or Mega Stone)
- **Tera type:** observed Tera narrows set candidates

**Deliverables:**
- [x] `observe_speed_comparison(our_speed, moved_first, trick_room_active)` on `OpponentBelief` — soft-penalises (×0.15) candidates whose effective speed (accounting for Choice Scarf) contradicts the observed turn order; ±2 tie threshold
- [x] `observe_damage_taken(base_power, is_special, attacker_stat, damage_fraction)` — soft-penalises (×0.20) candidates where expected damage fraction is >2× off; accounts for Assault Vest on SpD
- [x] `_candidate_effective_speed` and `_candidate_expected_damage_frac` helper functions
- [x] 22 tests in `test_opponent_belief.py` covering: helpers, scarf/AV detection, Trick Room inversion, speed-tie no-op, normalisation, edge cases (all passing)

---

## 4. Belief-Conditioned Value Training [x]

**Goal:** Teach the value model that high belief entropy = lower confidence, so it
hedges predictions when opponent team is uncertain.

**Why:** Currently the model sees expected feature values for unrevealed slots and
treats them as if certain. A mon that's "expected to be a sweeper" is not the same
as a confirmed sweeper — the model should widen its output distribution.

**Approach:**
- Add `belief_entropy` per opponent slot as an explicit input feature (already tracked
  in `TurnRecord`, just not passed to the value model)
- During training, weight loss by `1 / (1 + mean_opp_entropy)` — early turns with
  high uncertainty contribute less signal
- Optionally: dual-head value model outputting (mean, variance) with uncertainty loss

**Deliverables:**
- [x] Belief entropy features wired into state encoder (item 1 included this — token index 23)
- [x] `entropy_certainty_weights(opp_entropy)` in `train_transformer.py` — per-sample weight `1/(1+mean_opp_entropy)`, normalized to mean=1
- [x] `_OPP_ENTROPY_INDICES` — precomputed indices `[167, 191, 215, 239, 263, 287]` for the 6 opponent-slot entropy values in state_flat
- [x] `build_dataset()` updated to return `opp_entropy` array (5th return value)
- [x] `split_by_battle()` updated to pass entropy through train/val split
- [x] `train()` updated to accept `ent_tr`, `ent_val`, `entropy_weight` flag; combines class-balance weights with certainty weights; prints entropy tier BCE breakdown at eval time
- [x] `entropy_bce()` evaluation function — val BCE/accuracy broken down by low/mid/high entropy tier
- [x] `--no-entropy-weight` CLI flag to disable (default: on)
- [x] `--ablation` CLI flag — trains second unweighted model on same splits, prints side-by-side Δacc / Δbce summary

---

## Notes

- Items 1 and 2 are tightly coupled — do them together.
- Item 3 is independent and can be done in parallel or before items 1/2.
- Item 4 depends on item 1 (needs entropy in the state encoding).
- All improvements should be backward-compatible: existing heuristic + MCTS pipeline
  continues to work unchanged while new models are trained and evaluated.
