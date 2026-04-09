"""
Train a policy model on turns.jsonl.

Objective: per-turn softmax cross-entropy (KL) between predicted distribution
and MCTS visit fractions.  Distribution-matching — fracs within a turn compete.

Feature vector per action (50 features):
  [state_vec (24)]        — eval_terms, same for all actions in a turn
  [action_features (26)]  — per-action: bp_norm, accuracy, ..., is_switch, heuristic_score

Usage:
    python3 bot/learning/train_policy.py [--data data/turns.jsonl] [--out data/policy_weights.npz]
    python3 bot/learning/train_policy.py --hidden 32   # MLP with 32 hidden units (default)
    python3 bot/learning/train_policy.py --hidden 0    # linear (no hidden layer)
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from action_features import ACTION_FEATURE_NAMES

_N_ACTION = len(ACTION_FEATURE_NAMES)  # 25


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict], List[str]]:
    """
    Returns:
        turns_X       — list of [n_actions × 49] arrays (one per turn), unnormalized
        turns_y       — list of [n_actions] frac arrays (sum ≈ 1)
        turns_meta    — list of dicts: {turn, n_actions, max_frac, max_q_abs, battle_id}
        feature_order — sorted list of 24 eval_terms key names
    """
    # Discover canonical eval_terms feature order from first non-empty record
    feature_order: List[str] = []
    for rec in records:
        et = rec.get("eval_terms", {})
        if et:
            feature_order = sorted(et.keys())
            break

    if not feature_order:
        raise ValueError("No non-empty eval_terms found in data.")

    turns_X: List[np.ndarray] = []
    turns_y: List[np.ndarray] = []
    turns_meta: List[Dict] = []

    # Group by (battle_id, turn) — should be 1 record per group
    groups: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for rec in records:
        key = (rec["battle_id"], rec["turn"])
        groups[key].append(rec)

    # turn_count in player.py is never reset between battles, so compute
    # within-game turn as (turn - min_turn + 1) per battle for phase bucketing.
    battle_min_turn: Dict[str, int] = {}
    for rec in records:
        bid = rec["battle_id"]
        t = rec["turn"]
        if bid not in battle_min_turn or t < battle_min_turn[bid]:
            battle_min_turn[bid] = t

    skipped_no_features = 0
    for (bid, turn), group in groups.items():
        rec = group[0]

        action_stats = rec.get("action_stats", [])
        if len(action_stats) < 2:
            continue  # no distribution to learn from a single action

        # State features (same for all actions in a turn)
        et = rec.get("eval_terms", {})
        state_vec = np.array([et.get(k, 0.0) for k in feature_order], dtype=np.float64)

        X_rows = []
        y_rows = []
        has_any_features = False

        for a in action_stats:
            feats_dict: Dict[str, float] = a.get("features", {})
            if feats_dict:
                has_any_features = True
            # Backfill heuristic_score from prior_raw for old records that predate the field
            if "heuristic_score" not in feats_dict and "prior_raw" in a:
                feats_dict = dict(feats_dict)  # don't mutate the original
                feats_dict["heuristic_score"] = float(a["prior_raw"])
            action_feat = np.array(
                [feats_dict.get(k, 0.0) for k in ACTION_FEATURE_NAMES],
                dtype=np.float64,
            )
            feat = np.concatenate([state_vec, action_feat])  # (49,)
            X_rows.append(feat)
            y_rows.append(float(a["frac"]))

        if not has_any_features:
            skipped_no_features += 1
            continue  # old record without action features — skip to avoid noisy zeros

        X = np.stack(X_rows)   # (n_actions, 49)
        y = np.array(y_rows)   # (n_actions,)

        y_sum = y.sum()
        if y_sum < 1e-9:
            continue
        y = y / y_sum

        max_frac = float(y.max())
        # Q values in the same row order as y — used for calibration metrics.
        q_values = [float(a.get("q", 0.0)) for a in action_stats]
        max_q_abs = float(max(abs(q) for q in q_values))
        # Prefer n_actions_legal from search_info (true legal count) over len(action_stats)
        # (MCTS may not visit every action, so visited count can undercount branching factor)
        si = rec.get("search_info", {})
        n_actions = si.get("n_actions_legal", len(action_stats))

        turns_X.append(X)
        turns_y.append(y)
        within_turn = turn - battle_min_turn.get(bid, turn) + 1
        turns_meta.append({
            "battle_id":  bid,
            "turn":       within_turn,   # within-game turn for phase bucketing
            "n_actions":  n_actions,
            "max_frac":   max_frac,
            "max_q_abs":  max_q_abs,
            "q_values":   q_values,   # aligned with y rows, for Q-regret metric
        })

    if skipped_no_features:
        print(f"  Skipped {skipped_no_features} turns with no action features (old records).")

    return turns_X, turns_y, turns_meta, feature_order


# ---------------------------------------------------------------------------
# Train / val split (by battle to avoid leakage)
# ---------------------------------------------------------------------------

def split_by_battle(
    turns_X: List[np.ndarray],
    turns_y: List[np.ndarray],
    turns_meta: List[Dict],
    val_frac: float = 0.20,
    seed: int = 42,
) -> Tuple[List, List, List, List, List, List]:
    """
    Split into train / val sets by battle_id so turns from the same battle
    don't appear in both sets (prevents subtle leakage).
    Returns (train_X, train_y, train_meta, val_X, val_y, val_meta).
    """
    battle_ids = sorted({m["battle_id"] for m in turns_meta})
    rng = random.Random(seed)
    rng.shuffle(battle_ids)
    n_val = max(1, int(len(battle_ids) * val_frac))
    val_battles = set(battle_ids[:n_val])

    train_X, train_y, train_meta = [], [], []
    val_X,   val_y,   val_meta   = [], [], []
    for X, y, m in zip(turns_X, turns_y, turns_meta):
        if m["battle_id"] in val_battles:
            val_X.append(X);   val_y.append(y);   val_meta.append(m)
        else:
            train_X.append(X); train_y.append(y); train_meta.append(m)

    return train_X, train_y, train_meta, val_X, val_y, val_meta


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------

def standardize(
    turns_X: List[np.ndarray],
    feat_mean: Optional[np.ndarray] = None,
    feat_std: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Z-score normalize. If feat_mean/feat_std are provided, use them (for val set).
    Otherwise compute from turns_X (for train set).
    Returns (normalized_turns_X, feat_mean, feat_std).
    """
    if feat_mean is None:
        all_rows = np.concatenate(turns_X, axis=0)   # (total_actions, n_features)
        feat_mean = all_rows.mean(axis=0)
        feat_std  = all_rows.std(axis=0).clip(min=1e-8)
    turns_X_norm = [(X - feat_mean) / feat_std for X in turns_X]
    return turns_X_norm, feat_mean, feat_std


# ---------------------------------------------------------------------------
# Sample weights
# ---------------------------------------------------------------------------

def compute_sample_weights(turns_meta: List[Dict]) -> List[float]:
    """
    Per-turn importance weights:
      - Fully decided  (max_frac ≥ 0.85 AND max_q_abs > 0.95): 0.3× — trivial, low signal
      - One condition  (max_frac ≥ 0.85  OR max_q_abs > 0.95): 0.6× — partial signal
      - Near-uniform   (max_frac < 0.30):                        0.4× — noisy, MCTS found no preference
      - Informative    (0.30 ≤ max_frac < 0.85):                 1.0× — moderate preference, best signal
      - Branching: sqrt(n_actions / 2), capped at 2.0 — sublinear so it doesn't dominate
      - Phase: 1.2× early (≤15), 1.0× mid (16–40), 0.8× late (>40)
    Weights are returned unnormalized; train() normalises them to mean=1.
    """
    weights = []
    for m in turns_meta:
        max_frac  = m.get("max_frac", 0.0)
        max_q_abs = m.get("max_q_abs", 0.0)
        n_actions = m.get("n_actions", 2)
        turn      = m.get("turn", 50)

        if max_frac >= 0.85 and max_q_abs > 0.95:
            decided = 0.3   # both conditions: trivially decided
        elif max_frac >= 0.85 or max_q_abs > 0.95:
            decided = 0.6   # one condition: somewhat decided
        elif max_frac < 0.30:
            decided = 0.4   # near-uniform: MCTS found no clear preference, low signal
        else:
            decided = 1.0   # moderate preference: most informative

        # Sublinear: sqrt(n/2) grows as ~1.0, 1.4, 1.7, 2.0 for n=2,4,8,∞
        branching = min(2.0, float(np.sqrt(n_actions / 2.0)))

        if turn <= 15:
            phase = 1.2
        elif turn <= 40:
            phase = 1.0
        else:
            phase = 0.8

        weights.append(decided * branching * phase)
    return weights


def report_weight_coverage(turns_meta: List[Dict], sample_weights: List[float]) -> None:
    """
    Print effective gradient mass split across phases and decided/contested turns.
    Catches cases where almost all gradient comes from one subset without noticing.
    """
    sw = np.array(sample_weights, dtype=np.float64)
    sw_norm = sw / sw.sum()

    decided_both  = 0.0
    decided_one   = 0.0
    near_uniform  = 0.0
    phase_mass: Dict[str, float] = {"early": 0.0, "mid": 0.0, "late": 0.0}

    for i, m in enumerate(turns_meta):
        max_frac  = m.get("max_frac", 0.0)
        max_q_abs = m.get("max_q_abs", 0.0)
        turn      = m.get("turn", 50)

        if max_frac >= 0.85 and max_q_abs > 0.95:
            decided_both += sw_norm[i]
        elif max_frac >= 0.85 or max_q_abs > 0.95:
            decided_one  += sw_norm[i]
        elif max_frac < 0.30:
            near_uniform += sw_norm[i]

        ph = "early" if turn <= 15 else ("mid" if turn <= 40 else "late")
        phase_mass[ph] += sw_norm[i]

    informative = 1.0 - decided_both - decided_one - near_uniform
    print(f"\n  Effective weight coverage:")
    print(f"    Fully decided   (both, 0.3×): {decided_both:.1%}")
    print(f"    Partly decided  (one,  0.6×): {decided_one:.1%}")
    print(f"    Near-uniform    (0.4×):        {near_uniform:.1%}")
    print(f"    Informative     (1.0×):        {informative:.1%}")
    for ph in ["early", "mid", "late"]:
        print(f"    Phase {ph:6s}: {phase_mass[ph]:.1%}")


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


def cross_entropy(p: np.ndarray, y: np.ndarray) -> float:
    return -float(np.sum(y * np.log(np.clip(p, 1e-12, 1.0))))


def grad_ce(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of cross-entropy w.r.t. logits: dL/d_logit = p - y"""
    return p - y


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU (element-wise), computed from pre-activation z."""
    return (z > 0).astype(np.float64)


def mlp_forward(
    X: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    X:  (n, d)
    Returns (logits (n,), h (n, hidden), z1 (n, hidden))
    """
    z1 = X @ W1 + b1       # (n, hidden)
    h  = relu(z1)           # (n, hidden)
    logits = h @ w2 + b2   # (n,)
    return logits, h, z1


def mlp_backward(
    X: np.ndarray,
    z1: np.ndarray,
    h: np.ndarray,
    d_logits: np.ndarray,
    W1: np.ndarray,
    w2: np.ndarray,
    l2: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns (dW1, db1, dw2, db2) including L2 regularization on W1 and w2.
    d_logits: (n,) — gradient of loss w.r.t. logits (= p - y for cross-entropy)
    """
    dw2 = h.T @ d_logits                            # (hidden,)
    db2 = float(d_logits.sum())
    d_h = np.outer(d_logits, w2) * relu_grad(z1)   # (n, hidden)
    dW1 = X.T @ d_h                                  # (d, hidden)
    db1 = d_h.sum(axis=0)                            # (hidden,)
    dW1 += l2 * W1
    dw2 += l2 * w2
    return dW1, db1, dw2, db2


# ---------------------------------------------------------------------------
# Training loop (Adam, one turn per step, optional sample weights)
# ---------------------------------------------------------------------------

def train(
    turns_X: List[np.ndarray],
    turns_y: List[np.ndarray],
    sample_weights: Optional[List[float]] = None,
    n_epochs: int = 100,
    lr: float = 3e-3,
    l2: float = 1e-4,
    seed: int = 42,
    hidden_size: int = 0,
) -> Tuple[Dict, float]:
    """
    Returns (params_dict, final_weighted_avg_loss).

    Uses Adam optimizer (beta1=0.9, beta2=0.999).

    params_dict keys:
      - linear (hidden_size == 0): {"w": (n_features,), "b": scalar}
      - MLP    (hidden_size > 0):  {"W1": (n_features, hidden), "b1": (hidden,),
                                    "w2": (hidden,), "b2": scalar}
    """
    rng = random.Random(seed)
    n_features = turns_X[0].shape[1]
    n_turns = len(turns_X)
    order = list(range(n_turns))

    # Normalise sample weights so their mean = 1 (keeps lr scale stable)
    if sample_weights is not None:
        sw = np.array(sample_weights, dtype=np.float64)
        sw = sw / sw.mean()
    else:
        sw = np.ones(n_turns, dtype=np.float64)

    is_mlp = hidden_size > 0

    # Adam hyperparameters
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    t = 0  # global step counter

    if is_mlp:
        # Xavier init for W1 (good for ReLU), zeros for w2 / biases
        std = float(np.sqrt(2.0 / n_features))
        rng_np = np.random.default_rng(seed)
        W1 = rng_np.normal(0.0, std, (n_features, hidden_size))
        b1 = np.zeros(hidden_size, dtype=np.float64)
        w2 = np.zeros(hidden_size, dtype=np.float64)
        b2 = 0.0
        # Adam moment estimates
        m_W1 = np.zeros_like(W1);  v_W1 = np.zeros_like(W1)
        m_b1 = np.zeros_like(b1);  v_b1 = np.zeros_like(b1)
        m_w2 = np.zeros_like(w2);  v_w2 = np.zeros_like(w2)
        m_b2 = 0.0;                v_b2 = 0.0
        print(f"Training MLP ({n_features}→{hidden_size}→1) on {n_turns} turns, {n_epochs} epochs [Adam lr={lr}] ...")
    else:
        w = np.zeros(n_features, dtype=np.float64)
        b = 0.0
        m_w = np.zeros_like(w);  v_w = np.zeros_like(w)
        m_b = 0.0;               v_b = 0.0
        print(f"Training linear model on {n_turns} turns, {n_features} features, {n_epochs} epochs [Adam lr={lr}] ...")

    avg_loss = float("nan")
    for epoch in range(n_epochs):
        rng.shuffle(order)
        total_loss = 0.0
        total_w = 0.0

        for i in order:
            X  = turns_X[i]   # (n_actions, n_features)
            y  = turns_y[i]   # (n_actions,)
            wi = float(sw[i])

            t += 1
            bc1 = 1.0 - beta1 ** t
            bc2 = 1.0 - beta2 ** t

            if is_mlp:
                logits, h, z1 = mlp_forward(X, W1, b1, w2, b2)
                p = softmax(logits)
                total_loss += wi * cross_entropy(p, y)
                total_w    += wi
                d_logits = wi * grad_ce(p, y)
                dW1, db1_, dw2, db2_ = mlp_backward(X, z1, h, d_logits, W1, w2, l2)

                m_W1 = beta1 * m_W1 + (1 - beta1) * dW1
                v_W1 = beta2 * v_W1 + (1 - beta2) * dW1 ** 2
                W1  -= lr * (m_W1 / bc1) / (np.sqrt(v_W1 / bc2) + eps_adam)

                m_b1 = beta1 * m_b1 + (1 - beta1) * db1_
                v_b1 = beta2 * v_b1 + (1 - beta2) * db1_ ** 2
                b1  -= lr * (m_b1 / bc1) / (np.sqrt(v_b1 / bc2) + eps_adam)

                m_w2 = beta1 * m_w2 + (1 - beta1) * dw2
                v_w2 = beta2 * v_w2 + (1 - beta2) * dw2 ** 2
                w2  -= lr * (m_w2 / bc1) / (np.sqrt(v_w2 / bc2) + eps_adam)

                m_b2 = beta1 * m_b2 + (1 - beta1) * db2_
                v_b2 = beta2 * v_b2 + (1 - beta2) * db2_ ** 2
                b2  -= lr * (m_b2 / bc1) / (np.sqrt(v_b2 / bc2) + eps_adam)
            else:
                logits = X @ w + b
                p = softmax(logits)
                total_loss += wi * cross_entropy(p, y)
                total_w    += wi
                d_logits = grad_ce(p, y)
                gw = wi * X.T @ d_logits + l2 * w
                gb = float((wi * d_logits).sum())

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

def _dist_stats(vals: List[float]) -> Dict:
    """Return mean, median, p90, and count for a list of floats."""
    if not vals:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "count": 0}
    a = sorted(vals)
    cnt = len(a)
    mean = sum(a) / cnt
    mid = cnt // 2
    median = a[mid] if cnt % 2 else (a[mid - 1] + a[mid]) / 2.0
    p90 = a[min(int(0.90 * cnt), cnt - 1)]
    return {"mean": mean, "median": median, "p90": p90, "count": cnt}


def evaluate_metrics(
    turns_X: List[np.ndarray],
    turns_y: List[np.ndarray],
    turns_meta: List[Dict],
    score_fn: Callable[[np.ndarray], np.ndarray],
) -> Dict:
    """
    score_fn: maps X (n_actions, n_features) → logits (n_actions,)

    Returns:
        topk_acc     — {1, 2, 3}: fraction of turns where model top-k includes MCTS argmax
        avg_kl       — mean KL(y_mcts || p_model) in nats
        q_regret     — distribution stats for (max_Q - E_p[Q]) / q_range, clipped Q in [-1,1]
                       Only computed for turns where q_range > 0.05.
        top2_mass_byY — mean mass on top-2 MCTS-visited actions (n_actions ≥ 3 only)
        top2_mass_byQ — mean mass on top-2 highest-Q actions (same conditions as q_regret)
        phases       — all metrics stratified by early/mid/late
    """
    topk_correct = {1: 0, 2: 0, 3: 0}
    total_kl = 0.0
    qr_vals:  List[float] = []
    t2y_vals: List[float] = []
    t2q_vals: List[float] = []
    # phase_data entries: dicts with per-turn computed values
    phase_data: Dict[str, List[Dict]] = {"early": [], "mid": [], "late": []}

    n_total = len(turns_X)

    for i, (X, y) in enumerate(zip(turns_X, turns_y)):
        logits = score_fn(X)
        p = softmax(logits)

        mcts_best = int(np.argmax(y))
        ranked = np.argsort(logits)[::-1]
        for k in [1, 2, 3]:
            if mcts_best in ranked[:k]:
                topk_correct[k] += 1

        kl = float(np.sum(y * np.log(np.clip(y, 1e-12, 1.0) / np.clip(p, 1e-12, 1.0))))
        total_kl += kl

        # Q-regret + top2-by-Q: clip Q to rollout bounds, use safe denominator
        q_vals_raw = turns_meta[i].get("q_values", [])
        qr = t2q = None
        if len(q_vals_raw) == len(y):
            q = np.clip(np.array(q_vals_raw, dtype=np.float64), -1.0, 1.0)
            q_range = float(q.max() - q.min())
            den = max(q_range, 1e-6)
            if q_range > 0.05:
                qr = float((q.max() - float(p @ q)) / den)
                qr_vals.append(qr)
                top2q_idx = np.argsort(q)[::-1][:2]
                t2q = float(p[top2q_idx].sum())
                t2q_vals.append(t2q)

        # Top-2 mass by MCTS visits: only informative for n ≥ 3
        t2y = None
        if len(y) >= 3:
            top2y_idx = np.argsort(y)[::-1][:2]
            t2y = float(p[top2y_idx].sum())
            t2y_vals.append(t2y)

        turn = turns_meta[i].get("turn", 50)
        ph = "early" if turn <= 15 else ("mid" if turn <= 40 else "late")
        phase_data[ph].append({
            "mcts_best": mcts_best, "ranked": ranked, "kl": kl,
            "qr": qr, "t2y": t2y, "t2q": t2q,
        })

    result: Dict = {
        "topk_acc":      {k: topk_correct[k] / n_total if n_total else 0.0 for k in [1, 2, 3]},
        "avg_kl":        total_kl / n_total if n_total else 0.0,
        "q_regret":      {**_dist_stats(qr_vals),  "total": n_total},
        "top2_mass_byY": {**_dist_stats(t2y_vals), "total": n_total},
        "top2_mass_byQ": {**_dist_stats(t2q_vals), "total": n_total},
        "n":             n_total,
        "phases":        {},
    }

    for ph, data in phase_data.items():
        if not data:
            continue
        ph_topk = {
            k: sum(1 for d in data if d["mcts_best"] in d["ranked"][:k]) / len(data)
            for k in [1, 2, 3]
        }
        ph_qr  = [d["qr"]  for d in data if d["qr"]  is not None]
        ph_t2y = [d["t2y"] for d in data if d["t2y"] is not None]
        ph_t2q = [d["t2q"] for d in data if d["t2q"] is not None]
        result["phases"][ph] = {
            "topk_acc":      ph_topk,
            "avg_kl":        sum(d["kl"] for d in data) / len(data),
            "q_regret":      {**_dist_stats(ph_qr),  "total": len(data)},
            "top2_mass_byY": {**_dist_stats(ph_t2y), "total": len(data)},
            "top2_mass_byQ": {**_dist_stats(ph_t2q), "total": len(data)},
            "n":             len(data),
        }

    return result


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def show_diagnostics(
    score_fn: Callable[[np.ndarray], np.ndarray],
    feature_order: List[str],
    val_X: List[np.ndarray],
    val_y: List[np.ndarray],
    val_meta: List[Dict],
    top_k: int = 8,
    w: Optional[np.ndarray] = None,
    b: Optional[float] = None,
    is_mlp: bool = False,
):
    n_state = len(feature_order)

    if not is_mlp and w is not None and b is not None:
        w_state  = w[:n_state]
        w_action = w[n_state:]

        print("\nTop state feature weights (|w|):")
        for rank, i in enumerate(np.argsort(np.abs(w_state))[::-1][:top_k]):
            print(f"  {rank+1:2d}. {feature_order[i]:<30}  w = {w_state[i]:+.4f}")

        print("\nTop action feature weights (|w|):")
        for rank, i in enumerate(np.argsort(np.abs(w_action))[::-1][:top_k]):
            print(f"  {rank+1:2d}. {ACTION_FEATURE_NAMES[i]:<30}  w = {w_action[i]:+.4f}")

        print(f"\n       bias                             b = {b:+.4f}")
    else:
        print("\n[MLP] Weight diagnostics not shown (hidden layer — no scalar feature weights).")

    def _fmt_regret(d: Dict) -> str:
        cnt, tot = d["count"], d["total"]
        if cnt == 0:
            return f"n/a  (valid 0/{tot})"
        return f"mean={d['mean']:.3f}  med={d['median']:.3f}  p90={d['p90']:.3f}  (valid {cnt}/{tot})"

    def _fmt_mass(d: Dict) -> str:
        cnt, tot = d["count"], d["total"]
        if cnt == 0:
            return f"n/a  (valid 0/{tot})"
        return f"mean={d['mean']:.3f}  (valid {cnt}/{tot})"

    # Validation metrics
    if val_X:
        m = evaluate_metrics(val_X, val_y, val_meta, score_fn)
        ta = m["topk_acc"]
        print(f"\nVal metrics  (n={m['n']:4d})  "
              f"top1={ta[1]:.3f}  top2={ta[2]:.3f}  top3={ta[3]:.3f}  "
              f"KL={m['avg_kl']:.4f}")
        print(f"  regret:   {_fmt_regret(m['q_regret'])}")
        print(f"  top2_byY: {_fmt_mass(m['top2_mass_byY'])}")
        print(f"  top2_byQ: {_fmt_mass(m['top2_mass_byQ'])}")
        for ph in ["early", "mid", "late"]:
            if ph in m["phases"]:
                pm = m["phases"][ph]
                pa = pm["topk_acc"]
                print(f"  {ph:6s} (n={pm['n']:4d})  "
                      f"top1={pa[1]:.3f}  top2={pa[2]:.3f}  top3={pa[3]:.3f}  "
                      f"kl={pm['avg_kl']:.4f}")
                print(f"    regret:   {_fmt_regret(pm['q_regret'])}")
                print(f"    top2_byY: {_fmt_mass(pm['top2_mass_byY'])}")
                print(f"    top2_byQ: {_fmt_mass(pm['top2_mass_byQ'])}")

    # Sample one val turn
    if val_X:
        import random as _rnd
        idx = _rnd.randrange(len(val_X))
        X, y = val_X[idx], val_y[idx]
        logits = score_fn(X)
        p = softmax(logits)
        is_switch_col = n_state + ACTION_FEATURE_NAMES.index("is_switch")
        turn = val_meta[idx].get("turn", "?")
        print(f"\nSample val turn (turn={turn}, {len(y)} actions):")
        print(f"  {'kind':>4}  {'MCTS frac':>9}  {'Predicted':>9}")
        for j in range(len(y)):
            kind = "sw" if X[j, is_switch_col] > 0.5 else "mv"
            print(f"  {kind:>4}  {y[j]:>9.3f}  {p[j]:>9.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",     default="data/turns.jsonl")
    p.add_argument("--out",      default="data/policy_weights.npz")
    p.add_argument("--epochs",   type=int,   default=100)
    p.add_argument("--lr",       type=float, default=3e-3)
    p.add_argument("--l2",       type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.20)
    p.add_argument("--hidden",   type=int,   default=32,
                   help="Hidden layer size for MLP (0 = linear model)")
    p.add_argument("--no-weights", action="store_true",
                   help="Disable sample weighting (uniform weights)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.data} ...")
    records = load_turns(args.data)
    print(f"  {len(records)} records loaded.")

    turns_X_raw, turns_y, turns_meta, feature_order = build_dataset(records)
    print(f"  {len(turns_X_raw)} turns usable.")

    if not turns_X_raw:
        print("No usable turns — collect more data first.")
        return

    # Train / val split by battle
    train_X_raw, train_y, train_meta, val_X_raw, val_y, val_meta = split_by_battle(
        turns_X_raw, turns_y, turns_meta, val_frac=args.val_frac,
    )
    print(f"  Train: {len(train_X_raw)} turns  |  Val: {len(val_X_raw)} turns")

    # Standardize using train statistics only; apply same transform to val
    train_X, feat_mean, feat_std = standardize(train_X_raw)
    val_X, _, _ = standardize(val_X_raw, feat_mean=feat_mean, feat_std=feat_std)

    # Sample weights
    if args.no_weights:
        sample_weights = None
        print("  Sample weighting: disabled")
    else:
        sample_weights = compute_sample_weights(train_meta)
        print("  Sample weighting: on")
        report_weight_coverage(train_meta, sample_weights)

    is_mlp = args.hidden > 0

    params, final_loss = train(
        train_X, train_y,
        sample_weights=sample_weights,
        n_epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        hidden_size=args.hidden,
    )

    # Build score_fn closure for diagnostics / metrics
    if is_mlp:
        W1 = params["W1"]
        b1 = params["b1"]
        w2 = params["w2"]
        b2 = params["b2"]
        def score_fn(X: np.ndarray) -> np.ndarray:
            z1 = X @ W1 + b1
            h  = relu(z1)
            return h @ w2 + b2
    else:
        w = params["w"]
        b = params["b"]
        def score_fn(X: np.ndarray) -> np.ndarray:
            return X @ w + b

    show_diagnostics(
        score_fn, feature_order, val_X, val_y, val_meta,
        w=params.get("w"), b=params.get("b"), is_mlp=is_mlp,
    )

    # Save
    save_kwargs = dict(
        feature_order=np.array(feature_order),
        feat_mean=feat_mean,
        feat_std=feat_std,
    )
    if is_mlp:
        save_kwargs.update(
            W1=W1,
            b1=b1,
            w2=w2,
            b2=np.array([b2]),
        )
    else:
        save_kwargs.update(
            w=w,
            b=np.array([b]),
        )

    np.savez(args.out, **save_kwargs)
    model_type = f"MLP ({args.hidden} hidden)" if is_mlp else "linear"
    print(f"\nSaved {model_type} → {args.out}  (final weighted avg loss: {final_loss:.4f})")


if __name__ == "__main__":
    main()
