# Roadmap: Belief-Aware MCTS + Learned Guidance (AlphaZero-Style)

## Context

The current system is a heuristic-guided MCTS for Gen 9 RandBats. It uses:
- `action_priors()` in `search.py:79-133` — softmax over heuristic move/switch scores
- `evaluate_leaf()` in `search.py:416` — single call to `evaluate_state()` (13-factor heuristic in `eval.py:1240`)
- `OpponentBelief` / `TeamBelief` in `opponent_model.py` — Bayesian role distributions already threaded through `ShadowState`

The goal is to replace heuristic priors + value function with a neural network that takes **belief summaries** as input and outputs `P(a|b)` + `V(b)`. MCTS still samples hidden states and reasons explicitly — learning guides planning, it does not replace it.

---

## Phase 0 — Instrumentation & Baseline (1–2 weeks)

**Goal:** Measurable baseline before touching anything.

**Deliverables:**
1. **Replay serializer** — hook into `player.py:battle_finished()` (line 177) to serialize per-turn records: `(belief_snapshot, legal_actions, mcts_visit_dist, root_q, chosen_action, outcome)`. The visit distribution already exists in `stats["top"]` from `search.py:578-616`.
2. **Self-play match driver** — script to run two `AdvancedHeuristicPlayer` instances to completion, save trajectories.
3. **Win rate evaluator** — K=200 games, win rate + confidence interval. Baseline vs itself = ~50%.
4. **Heuristic calibration check** — correlate `evaluate_state()` output at each ply with final game outcome. Expect ordinal correlation but poor absolute calibration. This motivates using MCTS-backed value targets in Phase 2.

**Code changes:** Only `player.py` (add trajectory logging). Everything else read-only.

---

## Phase 1 — Belief Featurization (2–3 weeks)

**Goal:** Fixed-size float vector representing belief state `b`, not raw game state `s`.

**New file:** `bot/mcts/belief_encoder.py` → `encode_belief_state(state: ShadowState) -> np.ndarray`

**Input schema (~352 features, flat float32):**

| Block | Features | Notes |
|---|---|---|
| Global context | 8 | ply/60, alive counts, weather one-hot, Trick Room, tailwinds |
| My active | 28 | HP, status, boosts, types, move role flags |
| My bench (5 slots) | 100 (5×20) | HP, alive, status, types, role flags; **stable team order** (slot index = team slot, not HP rank) |
| Opp active belief | 42 | Actual HP/status + **E[is_physical], E[has_setup], E[has_priority], E[speed_mult], E[physical_threat]** from `OpponentBelief.dist`; belief entropy; E[type], P(HDB), P(choice item) |
| Opp bench (5 slots) | 150 (5×30) | **Stable team slot order.** `is_revealed` flag per slot. Revealed: same belief summary. Unseen: marginalized from `TeamBelief.pool` (pre-cached per-species) |
| Side conditions | 24 | Hazard layers, screen turns (our + opp sides), `opp_avg_boots_prob` |

**Reuse existing belief utilities:**
- `belief_certainty()` in `eval.py:220`
- `_boots_prob_for_belief()` in `eval.py:163`
- `_per_mon_threat_score()` in `eval.py:261` — already computes `E[has_setup]`, `E[has_priority]`, `E[speed_mult]`, `E[physical_threat]`

**Encoding note:** Do not sort bench slots by HP. HP chip every turn reshuffles ordering, making "slot 2" mean different things across turns and breaking identity continuity. Stable team-slot order preserves identity; the `is_revealed` flag handles the known/unknown distinction. True permutation invariance (SetTransformer) is Phase 5 territory.

**Milestone:** Encoder produces valid vectors for 1000+ states from saved games. No NaN/inf, all features in expected ranges.

**Code changes:** New file only. Nothing else touched.

---

## Phase 1.5 — Determinization Correctness Tests (3–5 days)

**Goal:** Verify that `determinize_opponent()` in `opponent_model.py` respects all observed constraints before any training happens. Training on belief states that violate observed game events produces garbage targets.

**Unit tests to write (`tests/test_determinization.py`):**
- If a move has been observed (`observe_move()` called), every sampled determinization includes that move in its move-4 set
- If an item has been revealed, every sampled determinization matches it
- If ability has been revealed, every sampled determinization matches it
- Speed constraints: sampled EV/nature combos don't produce speed stats that contradict observed speed ordering (e.g., opponent moved first under Tailwind — sampled speed must exceed ours)
- `TeamBelief` without-replacement sampling: same species never appears twice in a single sampled team
- After `observe_move()` eliminates all candidates for a role, the belief doesn't crash — it either falls back gracefully or raises a detectable error

**Why this phase is non-optional:** If these constraints are silently violated during self-play data collection (Phase 4), the value and policy targets are computed from states that couldn't exist in reality. The network learns to exploit impossible information. This is subtle and hard to debug after training has started.

**Code changes:** New test file only.

---

## Phase 2 — Supervised Pre-Training (3–4 weeks)

**Goal:** Warm-start a network on existing heuristic MCTS trajectories before self-play.

**Architecture** (`bot/training/model.py`):
```python
class BeliefNet(nn.Module):
    # Shared trunk: Linear(352->256) -> LayerNorm -> ReLU x3
    # Policy head: Linear -> 20 action slots (masked softmax)
    # Value head: Linear -> 64 -> ReLU -> Linear -> 1 -> Tanh
```
20 action slots covers max 4 moves + 5 switches + buffer. Actions sorted canonically; illegal actions masked in loss.

**Training targets:**
- **Policy:** MCTS visit distribution renormalized over legal actions (AlphaZero-style)
- **Value:** `0.7 x game_outcome + 0.3 x root.Q` (blended; avoid uncalibrated raw heuristic)

**Data:** Run 2000–5000 games with current heuristic MCTS, store trajectories. Replay buffer = `deque(maxlen=500_000)`. Shuffle across games (turns within a game are correlated).

**Loss:** `CrossEntropy(policy_logits[legal_mask], visit_dist) + MSE(value, target) + 1e-4 L2`

**Milestone:** Policy top-1 accuracy >50% on held-out positions. Value Pearson r > 0.4 vs game outcome.

**New files:** `bot/training/model.py`, `bot/training/train.py`, `bot/training/replay_buffer.py`

---

## Phase 3 — NN-Guided MCTS Integration (3–4 weeks)

**Goal:** Plug the NN into the two injection points in `search.py` with full backward compatibility.

**Integration points:**

1. **`action_priors()` (`search.py:79`)** — add optional `nn_model` + `nn_weight` params:
   ```python
   if nn_model and nn_weight > 0:
       nn_priors = nn_model(encode_belief_state(state)).policy
       return blend_priors(heuristic_priors, nn_priors, nn_weight)
   ```
   `nn_weight=0.0` (default) = unchanged behavior.

2. **`evaluate_leaf()` (`search.py:416`)** — add optional `nn_model` + `nn_value_weight`:
   ```python
   h_val = evaluate_state(node.state)
   if nn_model and nn_value_weight > 0:
       nn_val = nn_model(encode_belief_state(node.state)).value
       return (1 - nn_value_weight) * h_val + nn_value_weight * nn_val
   ```

3. **`MCTSConfig` (`search.py:12`)** — add fields:
   ```python
   nn_model: Optional[Any] = None
   nn_policy_weight: float = 0.0
   nn_value_weight: float = 0.0
   ```

4. **`player.py`** — pass `MCTSConfig(nn_model=loaded_net, ...)` to `mcts_pick_action()`.

**Deployment sequence:**
- Week 1: policy only (`nn_policy_weight=0.5`, `nn_value_weight=0.0`) — measure win rate
- Week 2: add value (`nn_value_weight=0.5`) — measure win rate
- Week 3: grid search (0.25, 0.5, 0.75, 1.0)^2 — find best operating point

**Belief sampling is non-optional:** Each simulation samples a determinization from the belief at the root via `determinize_opponent()` before traversal (POMCP-style). The opponent's hidden team, moves, and item are pinned for that simulation. On opponent-reveal events mid-game (e.g., a new move is used), the belief is updated via `observe_move()` before the next root search. This is what makes the search "belief-aware" — not just the NN input, but the search itself reasons over sampled worlds. This behavior already exists in the current system; the integration phase must not break it.

**Caching:** Only call NN at root node for policy (highest leverage, cheapest). Optionally cache leaf value calls by state hash.

**Milestone:** Statistically significant win rate improvement over pure heuristic MCTS (target p<0.05 over 500 games). Decision latency <5s per turn.

---

## Phase 4 — Self-Play Training Loop (6–8 weeks)

**Goal:** AlphaZero training loop — generate data with NN-guided MCTS, train on MCTS-backed targets, repeat.

**Architecture:**
```
Self-Play Worker -> Replay Buffer (deque 1M) -> Network Trainer
     ^_______________ checkpoint every 500 steps _______________^
```

**Key decisions:**

| Decision | Choice | Rationale |
|---|---|---|
| Both sides use same NN? | Yes | RandBats symmetry, 2x more training signal |
| Opponent MCTS budget | Phase 4: direct NN sampling (no MCTS). Phase 5: N=50 | Speed vs quality tradeoff |
| Temperature | 1.0 for first 20 moves, 0.0 rest | Exploration then exploitation |
| Dirichlet noise | `alpha=0.3–1.0`, `eps=0.25` at root | AlphaZero exploration mechanism (currently 0 in MCTSConfig) |
| Value target | `0.7 x outcome + 0.3 x V_NN(b_{t+3})` (3-step bootstrap) | Reduces variance of sparse reward |

**Belief encoding during self-play:** Use `perspective` param in encoder — our side sees our beliefs + uncertainty about opponent; opponent side sees mirror image. Beliefs updated only from *observations* (same as real play) — never give perfect information.

**New file:** `bot/training/self_play.py`

**Possible addition to `shadow_state.py`:** `step_from_both_actions(our_action, opp_action)` to make the game loop cleaner (currently `step()` internally samples the opponent).

**Training cadence:** 50 games -> 200 gradient steps (batch 256) -> repeat. Evaluate vs heuristic baseline every 500 steps. LR: 1e-3 -> 1e-4 -> 1e-5 (step decay).

**Milestone:** Learning curve showing win rate vs training steps. Value calibration r > 0.6 vs game outcome. Policy entropy decreasing without collapse.

---

## Phase 5 — Architecture & Curriculum Improvements (4–6 weeks, optional)

Only invest if Phase 4 plateaus before target win rate.

- **SetTransformer** over 12 Pokemon slots (6 mine + 6 opp) for proper permutation invariance. Only if MLP bottleneck is confirmed by ablation.
- **Curriculum:** N=50 -> 100 -> 200 sims as training progresses; switch opponent from direct NN to N=50 MCTS.
- **Information velocity features** in encoder: `n_moves_revealed / 4`, `roles_eliminated / initial_roles`.
- **Opponent value head:** dual output `V_me`, `V_opp` enforcing zero-sum structure.

---

## Phase 6 — Production Hardening (ongoing)

- Fallback logic in `player.py`: if NN outputs NaN or times out, use heuristic-only MCTS.
- Batched leaf evaluation (collect all expanded leaves, one forward pass before backup).
- `ENCODER_VERSION` stored with checkpoints — refuse to load version mismatch.
- Model versioning with eval win rates logged per checkpoint.

---

## Critical Dependencies

```
Phase 0 (data infrastructure)
  └-> Phase 2 (need trajectory data)
Phase 1 (belief encoder)
  └-> Phase 2, 3, 4 (NN input)
Phase 1.5 (determinization tests)  <-- non-optional gate
  └-> Phase 2, 4 (training on valid belief states only)
Phase 2 (pre-trained weights)
  └-> Phase 3 (warm start, not cold)
Phase 3 (integration hooks)
  └-> Phase 4 (search uses NN)
Phase 4 (self-play loop)
  └-> Phase 5 (only if needed)
```

**Early stopping value:** Each phase is independently useful. Phase 3 alone gives NN-guided MCTS. Phase 4 gives a self-improving system.

---

## Critical Files

| File | Role in Roadmap |
|---|---|
| `bot/mcts/search.py:79-133` | `action_priors()` — policy injection point |
| `bot/mcts/search.py:416` | `evaluate_leaf()` — value injection point |
| `bot/mcts/search.py:12-46` | `MCTSConfig` — add NN fields |
| `bot/mcts/eval.py:163,220,261` | Existing belief utilities to reuse in encoder |
| `bot/mcts/eval.py:1240-1565` | `evaluate_state()` — what NN value head replaces |
| `bot/mcts/shadow_state.py:400-469` | `ShadowState` fields — featurization source |
| `bot/model/opponent_model.py` | `OpponentBelief.dist`, `SetCandidate`, `TeamBelief.pool` |
| `bot/player.py:177` | `battle_finished()` — trajectory serialization hook |
| `bot/player.py:86` | `mcts_pick_action()` — pass NN config here |

---

## Verification at Each Phase

- **Phase 0:** 200 self-play games complete, trajectories serialized, baseline win rate measured
- **Phase 1:** Encoder runs on 1000+ states without NaN/inf; feature distributions look reasonable
- **Phase 1.5:** All determinization unit tests pass; no constraint violations found in 10k sampled determinizations
- **Phase 2:** Policy top-1 accuracy >50%, value r>0.4 on held-out data
- **Phase 3:** Win rate significantly above heuristic baseline; decision latency <5s
- **Phase 4:** Learning curve (win rate vs training steps); value r>0.6; policy entropy stable
- **Phase 5:** Win rate vs heuristic >60% sustained over 2000 games
