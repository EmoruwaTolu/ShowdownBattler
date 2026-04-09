"""
Train the TransformerValueModel on turn records that include a `state_flat` field.

The `state_flat` field is a 312-element list produced by encode_state_flat() and
logged by DataCollector when a ShadowState is available at decision time.

Training signal
  outcome == +1  → y = 1.0  (win)
  outcome == -1  → y = 0.0  (loss)
  outcome ==  0  → skipped  (tie / unknown)

Belief-conditioned loss weighting (item 4)
  Each sample is additionally weighted by 1 / (1 + mean_opp_entropy), where
  mean_opp_entropy is the average belief_entropy across the 6 opponent slots in
  state_flat. This down-weights early turns where the opponent team is still mostly
  unknown, so the model learns more from high-certainty decision points.
  Disable with --no-entropy-weight.

Usage
-----
  python3 bot/learning/train_transformer.py
  python3 bot/learning/train_transformer.py --data data/turns.jsonl --out data/transformer_weights.npz
  python3 bot/learning/train_transformer.py --epochs 50 --lr 3e-4 --batch 64
  python3 bot/learning/train_transformer.py --no-entropy-weight   # disable entropy weighting
  python3 bot/learning/train_transformer.py --ablation            # also train unweighted model, compare
  python3 bot/learning/train_transformer.py --compare data/value_weights.npz
    # Evaluates the existing MLP model on the same val split for comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# Make sure the project root is on sys.path when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bot.learning.transformer_value import TransformerValueModel
from bot.learning.state_encoder import N_TOTAL, N_MON_FEATURES
from bot.learning.state_encoder import MON_FEATURE_NAMES

# Indices of belief_entropy within the flat state vector for each opponent slot (6–11).
# Each token is N_MON_FEATURES wide; belief_entropy is at its named position.
_BELIEF_ENTROPY_OFFSET: int = MON_FEATURE_NAMES.index("belief_entropy")
_OPP_ENTROPY_INDICES: List[int] = [
    (6 + i) * N_MON_FEATURES + _BELIEF_ENTROPY_OFFSET
    for i in range(6)
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(path: str) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_dataset(
    records: List[dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[int]]:
    """
    Returns (X, y, opp_entropy, battle_ids, turns).

    X           : (N, N_TOTAL) float64 — from the `state_flat` field
    y           : (N,) binary labels
    opp_entropy : (N,) float64 — mean belief_entropy over the 6 opponent slots
    battle_ids  : (N,) list of battle id strings (for train/val split)
    turns       : (N,) turn numbers
    """
    X_rows:   List[np.ndarray] = []
    y_rows:   List[float]      = []
    ent_rows: List[float]      = []
    bids:     List[str]        = []
    turns:    List[int]        = []
    skipped_no_flat = 0
    skipped_tie     = 0

    seen: set = set()
    for rec in records:
        outcome = int(rec.get("outcome", 0))
        if outcome == 0:
            skipped_tie += 1
            continue

        sf = rec.get("state_flat")
        if sf is None:
            skipped_no_flat += 1
            continue

        key = (rec.get("battle_id", ""), rec.get("turn", 0))
        if key in seen:
            continue
        seen.add(key)

        x = np.array(sf, dtype=np.float64)
        if x.shape[0] != N_TOTAL:
            skipped_no_flat += 1
            continue

        # Mean belief_entropy over the 6 opponent slots (indices pre-computed above).
        mean_opp_ent = float(np.mean(x[_OPP_ENTROPY_INDICES]))

        X_rows.append(x)
        y_rows.append(1.0 if outcome == 1 else 0.0)
        ent_rows.append(mean_opp_ent)
        bids.append(str(rec.get("battle_id", "")))
        turns.append(int(rec.get("turn", 0) or 0))

    if skipped_tie:
        print(f"  Skipped {skipped_tie} turns with unknown/tie outcome.")
    if skipped_no_flat:
        print(f"  Skipped {skipped_no_flat} turns without state_flat (old format).")

    if not X_rows:
        raise ValueError(
            "No usable records found with `state_flat` field.\n"
            "Run self-play with the new DataCollector to collect data first."
        )

    X   = np.stack(X_rows)
    y   = np.array(y_rows,   dtype=np.float64)
    ent = np.array(ent_rows, dtype=np.float64)
    n_win = int(y.sum())
    print(f"  {len(y)} labeled turns: {n_win} wins ({n_win/len(y):.1%}), "
          f"{len(y)-n_win} losses ({(len(y)-n_win)/len(y):.1%})")
    mean_e, med_e = float(ent.mean()), float(np.median(ent))
    print(f"  Opp belief entropy: mean={mean_e:.3f}  median={med_e:.3f}  "
          f"(0=fully known, 1=fully unknown)")
    return X, y, ent, bids, turns


def split_by_battle(
    X: np.ndarray,
    y: np.ndarray,
    ent: np.ndarray,
    bids: List[str],
    turns: List[int],
    val_frac: float = 0.20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           List[int], List[int]]:
    """Split by battle id to avoid leakage."""
    all_battles = sorted(set(bids))
    rng = random.Random(seed)
    rng.shuffle(all_battles)
    val_battles = set(all_battles[:max(1, int(len(all_battles) * val_frac))])

    tr  = [i for i, b in enumerate(bids) if b not in val_battles]
    val = [i for i, b in enumerate(bids) if b in val_battles]

    turns_arr = np.array(turns, dtype=np.int32)
    return (X[tr], y[tr], ent[tr], X[val], y[val], ent[val],
            turns_arr[tr].tolist(), turns_arr[val].tolist())


def class_balanced_weights(y: np.ndarray) -> np.ndarray:
    n_win  = float(y.sum())
    n_loss = float(len(y) - n_win)
    w = np.where(y == 1.0,
                 len(y) / (2.0 * max(n_win, 1.0)),
                 len(y) / (2.0 * max(n_loss, 1.0)))
    return w / w.mean()


def entropy_certainty_weights(opp_entropy: np.ndarray) -> np.ndarray:
    """
    Per-sample weight based on how well-known the opponent team is.

    w_i = 1 / (1 + mean_opp_entropy_i)

    Turns where the opponent is mostly unseen (entropy near 1) get weight ~0.5.
    Turns where the opponent is fully revealed (entropy near 0) get weight ~1.0.
    Weights are normalized to mean=1 so they don't inflate/deflate the overall
    learning rate relative to the class-balance weights.
    """
    w = 1.0 / (1.0 + np.clip(opp_entropy, 0.0, 1.0))
    mean_w = w.mean()
    if mean_w > 1e-9:
        w = w / mean_w
    return w


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def evaluate(
    model: TransformerValueModel,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 256,
    label: str = "Val",
) -> Dict:
    probs_list = []
    for start in range(0, len(X), batch_size):
        batch = X[start:start + batch_size]
        p, _ = model._forward(batch, return_cache=False)
        probs_list.append(p)
    probs = np.concatenate(probs_list)

    p_clip = np.clip(probs, 1e-12, 1.0 - 1e-12)
    bce  = float(-np.mean(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip)))
    acc  = float(((probs >= 0.5) == y).mean())

    print(f"\n{label}  (n={len(y)})")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  BCE loss : {bce:.4f}")

    bins = np.linspace(0.0, 1.0, 6)
    print("  Calibration (predicted → actual win rate):")
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        cnt  = int(mask.sum())
        if cnt:
            actual = float(y[mask].mean())
            pred   = float(probs[mask].mean())
            bar    = "█" * int(actual * 20)
            print(f"    {lo:.1f}-{hi:.1f}  n={cnt:4d}  pred={pred:.2f}  actual={actual:.2f}  {bar}")

    return {"bce": bce, "accuracy": acc, "n": len(y)}


def phase_bce(
    model: TransformerValueModel,
    X: np.ndarray,
    y: np.ndarray,
    turns: List[int],
    batch_size: int = 256,
) -> None:
    probs_list = []
    for start in range(0, len(X), batch_size):
        p, _ = model._forward(X[start:start + batch_size], return_cache=False)
        probs_list.append(p)
    probs = np.concatenate(probs_list)
    p_clip = np.clip(probs, 1e-12, 1.0 - 1e-12)
    t = np.array(turns, dtype=np.int32)

    print("\nVal BCE by turn phase:")
    for label, mask in [("early(1-6)",  (t >= 1)  & (t <= 6)),
                        ("mid(7-15)",   (t >= 7)  & (t <= 15)),
                        ("late(16+)",   (t >= 16))]:
        cnt = int(mask.sum())
        if cnt:
            bce = float(-np.mean(y[mask] * np.log(p_clip[mask]) +
                                 (1 - y[mask]) * np.log(1 - p_clip[mask])))
            print(f"  {label:<10} n={cnt:4d}  bce={bce:.4f}")


def entropy_bce(
    model: TransformerValueModel,
    X: np.ndarray,
    y: np.ndarray,
    opp_entropy: np.ndarray,
    batch_size: int = 256,
) -> None:
    """Val BCE broken down by opponent belief entropy tier."""
    probs_list = []
    for start in range(0, len(X), batch_size):
        p, _ = model._forward(X[start:start + batch_size], return_cache=False)
        probs_list.append(p)
    probs = np.concatenate(probs_list)
    p_clip = np.clip(probs, 1e-12, 1.0 - 1e-12)

    print("\nVal BCE by opponent belief entropy tier:")
    for label, lo, hi in [("low  (0.0-0.33)", 0.00, 0.333),
                           ("mid  (0.33-0.67)", 0.333, 0.667),
                           ("high (0.67-1.0)",  0.667, 1.001)]:
        mask = (opp_entropy >= lo) & (opp_entropy < hi)
        cnt = int(mask.sum())
        if cnt:
            bce = float(-np.mean(y[mask] * np.log(p_clip[mask]) +
                                 (1 - y[mask]) * np.log(1 - p_clip[mask])))
            acc = float(((probs[mask] >= 0.5) == y[mask]).mean())
            print(f"  {label}  n={cnt:4d}  bce={bce:.4f}  acc={acc:.3f}")


def compare_mlp(
    mlp_path: str,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> None:
    """Run the existing MLP value model on the val set for comparison."""
    try:
        from bot.learning.value_model import ValueModel
        mlp = ValueModel.load(mlp_path)
        # The MLP expects eval_terms dict; we can't reconstruct that from state_flat.
        # Instead, use the first N features if they overlap — just report a note.
        print("\n[MLP comparison] The existing MLP uses eval_terms features (different input).")
        print("  To compare fairly, collect new data with both models active.")
    except Exception as e:
        print(f"\n[MLP comparison] Could not load {mlp_path}: {e}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: TransformerValueModel,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    ent_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    ent_val: np.ndarray,
    turns_val: List[int],
    n_epochs: int = 40,
    lr: float = 3e-4,
    l2: float = 1e-4,
    batch_size: int = 64,
    balance: bool = True,
    entropy_weight: bool = True,
    eval_every: int = 5,
    label: str = "Val",
) -> Dict:
    """
    Train the model and return val metrics.

    entropy_weight : if True, multiply class-balance weights by certainty weights
                     (1 / (1 + mean_opp_entropy)), down-weighting turns where the
                     opponent team is poorly known.
    """
    n  = len(X_tr)
    sw = class_balanced_weights(y_tr) if balance else np.ones(n)
    if entropy_weight:
        sw = sw * entropy_certainty_weights(ent_tr)
        sw = sw / sw.mean()   # keep total scale stable after multiplication

    idx = np.arange(n)
    rng = np.random.default_rng(0)

    weight_desc = "class-balance + entropy-certainty" if entropy_weight else "class-balance only"
    print(f"\nTraining TransformerValueModel for {n_epochs} epochs "
          f"(batch={batch_size}, lr={lr}, l2={l2}) ...")
    print(f"  Sample weighting: {weight_desc}")
    if balance:
        n_win = int(y_tr.sum())
        print(f"  Class balancing on  (wins={n_win}, losses={n-n_win})")
    if entropy_weight:
        print(f"  Entropy weighting: mean_opp_ent={ent_tr.mean():.3f}  "
              f"weight range [{sw.min():.3f}, {sw.max():.3f}]")

    for epoch in range(1, n_epochs + 1):
        rng.shuffle(idx)
        total_loss = total_w = 0.0

        for start in range(0, n, batch_size):
            bi  = idx[start:start + batch_size]
            bsw = sw[bi]
            loss, grads = model.forward_backward(X_tr[bi], y_tr[bi], bsw)
            model.adam_step(grads, lr=lr, l2=l2)
            total_loss += loss * len(bi)
            total_w    += len(bi)

        avg_loss = total_loss / total_w
        if epoch % eval_every == 0 or epoch == n_epochs:
            print(f"  Epoch {epoch:3d}/{n_epochs}  train_loss={avg_loss:.4f}")

    metrics = evaluate(model, X_val, y_val, label=label)
    phase_bce(model, X_val, y_val, turns_val)
    entropy_bce(model, X_val, y_val, ent_val)
    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default="data/turns.jsonl")
    p.add_argument("--out",        default="data/transformer_weights.npz")
    p.add_argument("--epochs",     type=int,   default=40)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--l2",         type=float, default=1e-4)
    p.add_argument("--batch",      type=int,   default=64)
    p.add_argument("--val-frac",   type=float, default=0.20)
    p.add_argument("--d-model",    type=int,   default=64)
    p.add_argument("--n-heads",    type=int,   default=4)
    p.add_argument("--n-layers",   type=int,   default=2)
    p.add_argument("--d-ff",       type=int,   default=128)
    p.add_argument("--d-out-hidden", type=int, default=32)
    p.add_argument("--no-balance", action="store_true")
    p.add_argument("--no-entropy-weight", action="store_true",
                   help="Disable belief-certainty loss weighting (item 4)")
    p.add_argument("--ablation",   action="store_true",
                   help="Also train an unweighted model and compare val metrics side-by-side")
    p.add_argument("--compare",    default=None,
                   help="Path to existing MLP value_weights.npz to compare against")
    p.add_argument("--eval-every", type=int, default=5,
                   help="Print val metrics every N epochs")
    return p.parse_args()


def _make_model(args) -> TransformerValueModel:
    return TransformerValueModel(
        d_model      = args.d_model,
        n_heads      = args.n_heads,
        n_layers     = args.n_layers,
        d_ff         = args.d_ff,
        d_out_hidden = args.d_out_hidden,
    )


def main():
    args = parse_args()

    print(f"Loading {args.data} ...")
    records = load_records(args.data)
    print(f"  {len(records)} records loaded.")

    X, y, ent, bids, turns = build_dataset(records)
    X_tr, y_tr, ent_tr, X_val, y_val, ent_val, turns_tr, turns_val = split_by_battle(
        X, y, ent, bids, turns, val_frac=args.val_frac
    )
    print(f"  Train: {len(y_tr)} turns  |  Val: {len(y_val)} turns")

    model = _make_model(args)

    n_params = (N_TOTAL * args.d_model
                + 2 * args.d_model            # side emb
                + args.n_layers * (
                    4 * args.d_model ** 2      # Wq Wk Wv Wo
                    + args.d_model * args.d_ff # W1
                    + args.d_ff * args.d_model # W2
                    + args.d_ff + args.d_model # biases
                    + 4 * args.d_model         # LN gammas/betas
                )
                + (args.d_model + N_TOTAL - 12 * 24) * args.d_out_hidden  # output head
                + args.d_out_hidden + 1)
    print(f"\nModel: d_model={args.d_model}, n_heads={args.n_heads}, "
          f"n_layers={args.n_layers}, d_ff={args.d_ff}  (~{n_params/1000:.0f}K params)")

    use_entropy_weight = not args.no_entropy_weight
    metrics_main = train(
        model, X_tr, y_tr, ent_tr, X_val, y_val, ent_val, turns_val,
        n_epochs       = args.epochs,
        lr             = args.lr,
        l2             = args.l2,
        batch_size     = args.batch,
        balance        = not args.no_balance,
        entropy_weight = use_entropy_weight,
        eval_every     = args.eval_every,
        label          = "Val (entropy-weighted)" if use_entropy_weight else "Val",
    )

    model.save(args.out)
    print(f"\nSaved → {args.out}")

    # Ablation: also train an unweighted model and compare side-by-side.
    if args.ablation and use_entropy_weight:
        print("\n" + "=" * 60)
        print("ABLATION: training unweighted model (same splits, same seed) ...")
        model_unw = _make_model(args)
        metrics_unw = train(
            model_unw, X_tr, y_tr, ent_tr, X_val, y_val, ent_val, turns_val,
            n_epochs       = args.epochs,
            lr             = args.lr,
            l2             = args.l2,
            batch_size     = args.batch,
            balance        = not args.no_balance,
            entropy_weight = False,
            eval_every     = args.eval_every,
            label          = "Val (unweighted)",
        )
        print("\n" + "=" * 60)
        print("Ablation summary:")
        print(f"  entropy-weighted  →  acc={metrics_main['accuracy']:.3f}  "
              f"bce={metrics_main['bce']:.4f}")
        print(f"  unweighted        →  acc={metrics_unw['accuracy']:.3f}  "
              f"bce={metrics_unw['bce']:.4f}")
        delta_acc = metrics_main['accuracy'] - metrics_unw['accuracy']
        delta_bce = metrics_main['bce']      - metrics_unw['bce']
        print(f"  Δacc={delta_acc:+.3f}  Δbce={delta_bce:+.4f}  "
              f"({'weighted better' if delta_bce < 0 else 'unweighted better or equal'})")

    if args.compare:
        compare_mlp(args.compare, X_val, y_val)


if __name__ == "__main__":
    main()
