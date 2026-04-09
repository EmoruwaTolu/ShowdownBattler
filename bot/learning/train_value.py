"""
Train a value model on turns.jsonl.

Objective: binary cross-entropy between predicted win probability and battle outcome.
For each turn, we predict P(win | state_features) using primitive state features only.

Training signal:  outcome == +1 → y = 1.0 (won)
                  outcome == -1 → y = 0.0 (lost)
                  outcome ==  0 → skipped  (tie / unknown)

Feature set: curated primitives only (heuristic composites excluded to avoid circularity).
All features (including belief_entropy) are read from eval_terms.

Usage:
    python3 bot/learning/train_value.py [--data data/turns.jsonl] [--out data/value_weights.npz]
    python3 bot/learning/train_value.py --hidden 16   # MLP with 16 hidden units (default)
    python3 bot/learning/train_value.py --hidden 0    # logistic regression (no hidden layer)
    python3 bot/learning/train_value.py --all-features  # use all eval_terms (debug only)
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Feature allowlist — clean primitive signals only.
# Dropped (heuristic composites): core_norm, team_term, numbers_term, race_term,
# switch_term, pivot_term, progress_term, ahead, tempo_penalty, sac_penalty,
# setup_early_pen, post_ko_pen, active_preserve, priority_revenge_term, sack_bonus.
# ---------------------------------------------------------------------------

FILTERED_FEATURES: List[str] = [
    # Resource
    "hp_advantage",
    "alive_advantage",
    "status_burden",
    "hazard_burden",
    # Tactical
    "active_matchup",
    "speed_control",
    "switch_safety",
    "boost_advantage",
    # Strategic
    "wincon_readiness",
    "defensive_cover",
    # Belief
    "belief_entropy",
    "belief_weighted_threat",
    "hidden_speed_risk",
]

# Roadmap feature sets (battle-tested order; avoid heuristic composites).
FEATURE_SETS: Dict[str, List[str]] = {
    "full_clean": [
        "hp_advantage",
        "alive_advantage",
        "status_burden",
        "hazard_burden",
        "active_matchup",
        "speed_control",
        "switch_safety",
        "boost_advantage",
        "wincon_readiness",
        "defensive_cover",
        "belief_entropy",
        "belief_weighted_threat",
        "hidden_speed_risk",
    ],
    "no_belief": [
        "hp_advantage",
        "alive_advantage",
        "status_burden",
        "hazard_burden",
        "active_matchup",
        "speed_control",
        "switch_safety",
        "boost_advantage",
        "wincon_readiness",
        "defensive_cover",
    ],
    "resource_only": [
        "hp_advantage",
        "alive_advantage",
        "status_burden",
        "hazard_burden",
    ],
    "resource_tactical": [
        "hp_advantage",
        "alive_advantage",
        "status_burden",
        "hazard_burden",
        "active_matchup",
        "speed_control",
        "switch_safety",
        "boost_advantage",
    ],
    "resource_tactical_strategic": [
        "hp_advantage",
        "alive_advantage",
        "status_burden",
        "hazard_burden",
        "active_matchup",
        "speed_control",
        "switch_safety",
        "boost_advantage",
        "wincon_readiness",
        "defensive_cover",
    ],
}


def _print_feature_stats(X_raw: np.ndarray, feature_order: List[str]) -> None:
    print("\nFeature summary (raw, before standardization):")
    for i, name in enumerate(feature_order):
        col = X_raw[:, i]
        print(
            f"  {name:<24} mean={col.mean():+8.4f}  std={col.std():8.4f}  "
            f"min={col.min():+8.4f}  max={col.max():+8.4f}"
        )


def _phase_slices(turns: List[int]) -> List[Tuple[str, np.ndarray]]:
    t = np.array(turns, dtype=np.int32)
    return [
        ("early(1-6)", (t >= 1) & (t <= 6)),
        ("mid(7-15)", (t >= 7) & (t <= 15)),
        ("late(16+)", (t >= 16)),
    ]


def _bce_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    p = sigmoid(logits)
    return float(bce_loss(p, y))


def _val_logits(X: np.ndarray, params: Dict, is_mlp: bool) -> np.ndarray:
    if is_mlp:
        W1, b1, w2, b2 = params["W1"], params["b1"], params["w2"], params["b2"]
        h = relu(X @ W1 + b1)
        return h @ w2 + b2
    return X @ params["w"] + params["b"]


def load_turns(path: str) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_dataset(
    records: List[dict],
    all_features: bool = False,
    feature_list: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], dict]:
    """
    Returns:
        X             — (n_turns, n_features) array of state features
        y             — (n_turns,) binary labels: 1.0=win, 0.0=loss
        feature_order — list of feature names (matches columns of X)

    When all_features=False (default), uses FILTERED_FEATURES or feature_list if provided.
    belief_entropy is read from the top-level record, not from eval_terms.
    When all_features=True, uses all eval_terms keys (debug/comparison only).
    """
    # Discover available eval_terms keys from first record
    available_et_keys: set = set()
    for rec in records:
        et = rec.get("eval_terms", {})
        if et:
            available_et_keys = set(et.keys())
            break

    if not available_et_keys:
        raise ValueError("No non-empty eval_terms found in data.")

    if all_features:
        # Use all eval_terms features (sorted for determinism), no belief_entropy
        feature_order = sorted(available_et_keys)
        use_belief_entropy = False
        print(f"  Using ALL {len(feature_order)} eval_terms features (--all-features mode).")
    else:
        chosen = list(feature_list) if feature_list is not None else FILTERED_FEATURES
        # Keep only features that are actually present in the data.
        feature_order = [f for f in chosen if f in available_et_keys]

        missing = [f for f in chosen if f not in available_et_keys]
        if missing:
            print(f"  WARNING: {len(missing)} feature(s) not found in data "
                  f"(will be skipped): {missing}")

        print(f"  Using {len(feature_order)} features from eval_terms.")

    # Deduplicate by (battle_id, turn) — keep first occurrence
    seen: set = set()
    X_rows: List[np.ndarray] = []
    y_rows: List[float] = []
    skipped_ties = 0
    turns_list: List[int] = []
    battle_ids_list: List[str] = []

    for rec in records:
        outcome = int(rec.get("outcome", 0))
        if outcome == 0:
            skipped_ties += 1
            continue

        key = (rec["battle_id"], rec["turn"])
        if key in seen:
            continue
        seen.add(key)

        et = rec.get("eval_terms", {})

        vals = [et.get(k, 0.0) for k in feature_order]

        x = np.array(vals, dtype=np.float64)
        y = 1.0 if outcome == 1 else 0.0

        X_rows.append(x)
        y_rows.append(y)
        turns_list.append(int(rec.get("turn", 0) or 0))
        battle_ids_list.append(str(rec.get("battle_id", "")))

    if skipped_ties:
        print(f"  Skipped {skipped_ties} turns with unknown/tie outcome.")

    if not X_rows:
        raise ValueError("No labeled turns found (all outcomes were 0).")

    X = np.stack(X_rows)
    y = np.array(y_rows, dtype=np.float64)

    n_win  = int(y.sum())
    n_loss = int(len(y) - n_win)
    print(f"  {len(y)} labeled turns: {n_win} wins ({n_win/len(y):.1%}), "
          f"{n_loss} losses ({n_loss/len(y):.1%})")

    meta = {"turns_list": turns_list, "battle_ids_list": battle_ids_list}
    return X, y, feature_order, meta

def split_by_battle(
    records: List[dict],
    X: np.ndarray,
    y: np.ndarray,
    meta: Optional[dict] = None,
    val_frac: float = 0.20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split X, y into train/val by battle_id.
    Returns X_tr, y_tr, X_val, y_val, turns_tr, turns_val.
    """
    battle_ids_list: List[str]
    turns_arr: List[int]
    if meta is not None:
        battle_ids_list = list(meta["battle_ids_list"])
        turns_arr = list(meta["turns_list"])
    else:
        # Fallback: rebuild from records (no turn info available)
        seen: set = set()
        battle_ids_list = []
        turns_arr = []
        for rec in records:
            if int(rec.get("outcome", 0)) == 0:
                continue
            key = (rec["battle_id"], rec["turn"])
            if key in seen:
                continue
            seen.add(key)
            battle_ids_list.append(rec["battle_id"])
            turns_arr.append(int(rec.get("turn", 0) or 0))

    all_battles = sorted(set(battle_ids_list))
    rng = random.Random(seed)
    rng.shuffle(all_battles)
    n_val = max(1, int(len(all_battles) * val_frac))
    val_battles = set(all_battles[:n_val])

    t = np.array(turns_arr, dtype=np.int32)
    train_idx = [i for i, bid in enumerate(battle_ids_list) if bid not in val_battles]
    val_idx   = [i for i, bid in enumerate(battle_ids_list) if bid in val_battles]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], t[train_idx], t[val_idx]

def standardize(
    X: np.ndarray,
    feat_mean: Optional[np.ndarray] = None,
    feat_std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if feat_mean is None:
        feat_mean = X.mean(axis=0)
        feat_std  = X.std(axis=0).clip(min=1e-8)
    return (X - feat_mean) / feat_std, feat_mean, feat_std

def class_balanced_weights(y: np.ndarray) -> np.ndarray:
    """
    Give each sample weight = 1 / (2 * class_freq), so wins and losses
    contribute equal total gradient mass regardless of class imbalance.
    Weights are normalised to mean=1.
    """
    n_win  = float(y.sum())
    n_loss = float(len(y) - n_win)
    w = np.where(y == 1.0, len(y) / (2.0 * max(n_win, 1.0)),
                            len(y) / (2.0 * max(n_loss, 1.0)))
    return w / w.mean()

def sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def bce_loss(p: np.ndarray, y: np.ndarray) -> float:
    """Binary cross-entropy (scalar)."""
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(np.float64)

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    n_epochs: int = 100,
    lr: float = 3e-3,
    l2: float = 1e-4,
    seed: int = 42,
    hidden_size: int = 0,
) -> Tuple[Dict, float]:
    """
    Returns (params_dict, final_weighted_avg_loss).

    params_dict keys:
      - logistic  (hidden_size == 0): {"w": (n_features,), "b": scalar}
      - MLP       (hidden_size > 0):  {"W1": (n_features, hidden), "b1": (hidden,),
                                       "w2": (hidden,), "b2": scalar}
    """
    rng = random.Random(seed)
    n, n_features = X_train.shape
    order = list(range(n))

    sw = sample_weights if sample_weights is not None else np.ones(n, dtype=np.float64)
    sw = sw / sw.mean()

    is_mlp = hidden_size > 0

    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    t = 0

    if is_mlp:
        std = float(np.sqrt(2.0 / n_features))
        rng_np = np.random.default_rng(seed)
        W1 = rng_np.normal(0.0, std, (n_features, hidden_size))
        b1 = np.zeros(hidden_size, dtype=np.float64)
        w2 = np.zeros(hidden_size, dtype=np.float64)
        b2 = 0.0
        m_W1 = np.zeros_like(W1); v_W1 = np.zeros_like(W1)
        m_b1 = np.zeros_like(b1); v_b1 = np.zeros_like(b1)
        m_w2 = np.zeros_like(w2); v_w2 = np.zeros_like(w2)
        m_b2 = 0.0; v_b2 = 0.0
        print(f"Training value MLP ({n_features}→{hidden_size}→1) on {n} turns, "
              f"{n_epochs} epochs [Adam lr={lr}] ...")
    else:
        w = np.zeros(n_features, dtype=np.float64)
        b = 0.0
        m_w = np.zeros_like(w); v_w = np.zeros_like(w)
        m_b = 0.0; v_b = 0.0
        print(f"Training logistic regression ({n_features}→1) on {n} turns, "
              f"{n_epochs} epochs [Adam lr={lr}] ...")

    avg_loss = float("nan")
    for epoch in range(n_epochs):
        rng.shuffle(order)
        total_loss = 0.0
        total_w    = 0.0

        for i in order:
            x  = X_train[i]   # (n_features,)
            yi = y_train[i]    # scalar 0 or 1
            wi = float(sw[i])

            t += 1
            bc1 = 1.0 - beta1 ** t
            bc2 = 1.0 - beta2 ** t

            if is_mlp:
                z1     = x @ W1 + b1             # (hidden,)
                h      = relu(z1)
                logit  = float(h @ w2 + b2)
                p      = float(sigmoid(np.array([logit]))[0])
                loss   = -(yi * np.log(max(p, 1e-12)) + (1 - yi) * np.log(max(1 - p, 1e-12)))
                total_loss += wi * loss
                total_w    += wi

                d_logit = wi * (p - yi)
                dw2 = d_logit * h + l2 * w2
                db2 = d_logit
                d_h = d_logit * w2 * relu_grad(z1)
                dW1 = np.outer(x, d_h) + l2 * W1
                db1 = d_h

                m_W1 = beta1 * m_W1 + (1 - beta1) * dW1
                v_W1 = beta2 * v_W1 + (1 - beta2) * dW1 ** 2
                W1  -= lr * (m_W1 / bc1) / (np.sqrt(v_W1 / bc2) + eps_adam)

                m_b1 = beta1 * m_b1 + (1 - beta1) * db1
                v_b1 = beta2 * v_b1 + (1 - beta2) * db1 ** 2
                b1  -= lr * (m_b1 / bc1) / (np.sqrt(v_b1 / bc2) + eps_adam)

                m_w2 = beta1 * m_w2 + (1 - beta1) * dw2
                v_w2 = beta2 * v_w2 + (1 - beta2) * dw2 ** 2
                w2  -= lr * (m_w2 / bc1) / (np.sqrt(v_w2 / bc2) + eps_adam)

                m_b2 = beta1 * m_b2 + (1 - beta1) * db2
                v_b2 = beta2 * v_b2 + (1 - beta2) * db2 ** 2
                b2  -= lr * (m_b2 / bc1) / (np.sqrt(v_b2 / bc2) + eps_adam)

            else:
                logit = float(x @ w + b)
                p     = float(sigmoid(np.array([logit]))[0])
                loss  = -(yi * np.log(max(p, 1e-12)) + (1 - yi) * np.log(max(1 - p, 1e-12)))
                total_loss += wi * loss
                total_w    += wi

                d_logit = wi * (p - yi)
                gw = d_logit * x + l2 * w
                gb = d_logit

                m_w = beta1 * m_w + (1 - beta1) * gw
                v_w = beta2 * v_w + (1 - beta2) * gw ** 2
                w  -= lr * (m_w / bc1) / (np.sqrt(v_w / bc2) + eps_adam)

                m_b = beta1 * m_b + (1 - beta1) * gb
                v_b = beta2 * v_b + (1 - beta2) * gb ** 2
                b  -= lr * (m_b / bc1) / (np.sqrt(v_b / bc2) + eps_adam)

        avg_loss = total_loss / total_w
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs}  weighted_avg_loss = {avg_loss:.4f}")

    if is_mlp:
        return {"W1": W1, "b1": b1, "w2": w2, "b2": b2}, avg_loss
    else:
        return {"w": w, "b": b}, avg_loss


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def evaluate_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict,
    is_mlp: bool,
) -> Dict:
    """
    Returns accuracy, mean BCE loss, and a calibration table.
    """
    n = len(y)
    if n == 0:
        return {}

    # Forward pass
    if is_mlp:
        W1, b1, w2, b2 = params["W1"], params["b1"], params["w2"], params["b2"]
        h = relu(X @ W1 + b1)
        logits = h @ w2 + b2
    else:
        logits = X @ params["w"] + params["b"]

    probs = sigmoid(logits)

    acc  = float(((probs >= 0.5) == y).mean())
    loss = float(bce_loss(probs, y))

    # Calibration: 5 bins
    bins = np.linspace(0.0, 1.0, 6)
    calib = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        cnt = int(mask.sum())
        if cnt > 0:
            actual = float(y[mask].mean())
            pred   = float(probs[mask].mean())
            calib.append((f"{lo:.1f}-{hi:.1f}", cnt, pred, actual))

    return {"accuracy": acc, "bce_loss": loss, "calibration": calib, "n": n}


def show_diagnostics(
    params: Dict,
    feature_order: List[str],
    X_val: np.ndarray,
    y_val: np.ndarray,
    is_mlp: bool,
    top_k: int = 8,
) -> None:
    if not is_mlp:
        w = params["w"]
        b = params["b"]
        print("\nTop feature weights (|w|) for win prediction:")
        for rank, i in enumerate(np.argsort(np.abs(w))[::-1][:top_k]):
            direction = "→ WIN" if w[i] > 0 else "→ LOSS"
            print(f"  {rank+1:2d}. {feature_order[i]:<30}  w = {w[i]:+.4f}  {direction}")
        print(f"\n       bias  b = {b:+.4f}  "
              f"(base win prob ≈ {1/(1+np.exp(-b)):.1%})")
    else:
        print("\n[MLP value model] No scalar feature weights (hidden layer).")

    if len(X_val) > 0:
        m = evaluate_metrics(X_val, y_val, params, is_mlp)
        print(f"\nVal metrics  (n={m['n']:4d})")
        print(f"  Accuracy : {m['accuracy']:.3f}")
        print(f"  BCE loss : {m['bce_loss']:.4f}")
        print(f"  Calibration (predicted → actual win rate):")
        for bucket, cnt, pred, actual in m["calibration"]:
            bar = "█" * int(actual * 20)
            print(f"    {bucket}  n={cnt:4d}  pred={pred:.2f}  actual={actual:.2f}  {bar}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",     default="data/turns.jsonl")
    p.add_argument("--out",      default="data/value_weights.npz")
    p.add_argument("--epochs",   type=int,   default=100)
    p.add_argument("--lr",       type=float, default=3e-3)
    p.add_argument("--l2",       type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.20)
    p.add_argument("--hidden",   type=int,   default=16,
                   help="Hidden layer size (0 = logistic regression)")
    p.add_argument("--no-balance", action="store_true",
                   help="Disable class-balanced sample weights")
    p.add_argument("--all-features", action="store_true",
                   help="Use all eval_terms features instead of the curated primitive subset")
    p.add_argument(
        "--feature-set",
        default="full_clean",
        choices=sorted(FEATURE_SETS.keys()),
        help="Named feature set to train on (roadmap ablations). Ignored if --all-features.",
    )
    p.add_argument(
        "--print-feature-stats",
        action="store_true",
        help="Print mean/std/min/max for each feature (raw, before standardization).",
    )
    p.add_argument(
        "--results-csv",
        default=None,
        help="If set, append a one-line CSV result summary for this run.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.data} ...")
    records = load_turns(args.data)
    print(f"  {len(records)} records loaded.")

    feature_list = None if args.all_features else FEATURE_SETS.get(args.feature_set, FILTERED_FEATURES)
    if not args.all_features:
        print(f"Feature set: {args.feature_set}  (n={len(feature_list)})")
    X_raw, y, feature_order, meta = build_dataset(records, all_features=args.all_features, feature_list=feature_list)
    print(f"  {len(y)} usable turns.")

    if len(y) == 0:
        print("No usable turns.")
        return

    if args.print_feature_stats and not args.all_features:
        _print_feature_stats(X_raw, feature_order)

    # Train / val split by battle
    X_tr_raw, y_tr, X_val_raw, y_val, turns_tr, turns_val = split_by_battle(
        records, X_raw, y, meta=meta, val_frac=args.val_frac
    )
    print(f"  Train: {len(y_tr)} turns  |  Val: {len(y_val)} turns")

    # Standardize
    X_tr, feat_mean, feat_std = standardize(X_tr_raw)
    X_val, _, _ = standardize(X_val_raw, feat_mean=feat_mean, feat_std=feat_std)

    # Class-balanced sample weights
    if args.no_balance:
        sw = None
        print("  Class balancing: disabled")
    else:
        sw = class_balanced_weights(y_tr)
        n_win  = int(y_tr.sum())
        n_loss = int(len(y_tr) - n_win)
        print(f"  Class balancing: on  (wins {n_win}, losses {n_loss}, "
              f"pos_weight ≈ {n_loss/max(n_win,1):.1f}×)")

    is_mlp = args.hidden > 0
    params, final_loss = train(
        X_tr, y_tr,
        sample_weights=sw,
        n_epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        hidden_size=args.hidden,
    )

    show_diagnostics(params, feature_order, X_val, y_val, is_mlp)

    # Phase-sliced val BCE (early/mid/late)
    logits_val = _val_logits(X_val, params, is_mlp=is_mlp)
    print("\nVal BCE by phase:")
    for label, mask in _phase_slices(list(turns_val)):
        cnt = int(mask.sum())
        if cnt <= 0:
            continue
        bce = _bce_from_logits(logits_val[mask], y_val[mask])
        print(f"  {label:<10} n={cnt:4d}  bce={bce:.4f}")

    # Save
    save_kwargs = dict(
        feature_order=np.array(feature_order),
        feat_mean=feat_mean,
        feat_std=feat_std,
    )
    if is_mlp:
        save_kwargs.update(W1=params["W1"], b1=params["b1"],
                           w2=params["w2"], b2=np.array([params["b2"]]))
    else:
        save_kwargs.update(w=params["w"], b=np.array([params["b"]]))

    np.savez(args.out, **save_kwargs)
    model_type = f"MLP ({args.hidden} hidden)" if is_mlp else "logistic regression"
    print(f"\nSaved {model_type} → {args.out}  (final loss: {final_loss:.4f})")

    # Append results row
    if args.results_csv:
        try:
            m = evaluate_metrics(X_val, y_val, params, is_mlp)
            row = {
                "experiment": (args.feature_set if not args.all_features else "all_features"),
                "n_features": len(feature_order),
                "model": ("mlp" if is_mlp else "logreg"),
                "hidden": int(args.hidden),
                "val_bce": float(m.get("bce_loss", float("nan"))),
                "val_acc": float(m.get("accuracy", float("nan"))),
                "n_val": int(m.get("n", 0)),
            }
            import csv
            import os

            write_header = not os.path.exists(args.results_csv)
            with open(args.results_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    w.writeheader()
                w.writerow(row)
            print(f"Appended results → {args.results_csv}")
        except Exception as _csv_err:
            print(f"[results_csv] failed ({_csv_err})")


if __name__ == "__main__":
    main()
