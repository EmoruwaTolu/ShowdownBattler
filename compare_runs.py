"""
compare_runs.py — Side-by-side A/B comparison of two selfplay JSONL files.

Usage:
    python3 compare_runs.py data/ab_heur.jsonl data/ab_learned.jsonl

Reports:
  - Win rate, game length
  - Search efficiency: max_visits_frac, q_range (overall + by phase)
  - Win/loss search profile (do winning games look different?)
  - Hard-state analysis: gains by q_range bucket and branching factor
  - Policy usage sanity check
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def load(path: str) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def phase(turn: int) -> str:
    if turn <= 15:
        return "early"
    if turn <= 40:
        return "mid"
    return "late"


def avg(lst: List[float]) -> float:
    return sum(lst) / len(lst) if lst else float("nan")


def _build_battle_index(records: List[dict]) -> Tuple[
    Dict[str, List[dict]],   # battles
    Dict[str, int],           # battle_min_turn
    Dict[str, int],           # battle_outcome
]:
    battles: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        battles[r["battle_id"]].append(r)
    battle_min_turn = {
        bid: min(t["turn"] for t in turns)
        for bid, turns in battles.items()
    }
    battle_outcome = {
        bid: turns[-1]["outcome"]
        for bid, turns in battles.items()
    }
    return battles, battle_min_turn, battle_outcome


def summarize(records: List[dict], label: str) -> None:
    battles, battle_min_turn, battle_outcome = _build_battle_index(records)

    # --- Battle-level stats ---
    n_battles = len(battles)
    wins = sum(1 for o in battle_outcome.values() if o == 1)
    losses = sum(1 for o in battle_outcome.values() if o == -1)
    ties = n_battles - wins - losses
    game_lengths = [
        max(t["turn"] for t in turns) - battle_min_turn[bid] + 1
        for bid, turns in battles.items()
    ]
    win_rate = wins / n_battles if n_battles else 0.0
    avg_len = avg(game_lengths)

    # --- Turn-level stats, with per-turn metadata ---
    by_phase: Dict[str, Dict[str, List[float]]] = {
        ph: {"mvf": [], "qr": []} for ph in ["early", "mid", "late"]
    }
    # Win/loss profiles: search stats split by battle outcome
    win_mvf:  List[float] = []
    loss_mvf: List[float] = []
    win_qr:   List[float] = []
    loss_qr:  List[float] = []

    # Hard-state buckets: (qr_bucket, branch_bucket) -> list of mvf values
    # qr_bucket:     "easy" (q_range >= 0.4), "hard" (q_range < 0.4)
    # branch_bucket: "low" (n_actions_legal <= 4), "high" (n_actions_legal > 4)
    hard_buckets: Dict[str, List[float]] = defaultdict(list)

    all_mvf: List[float] = []
    all_qr:  List[float] = []
    policy_flags: List[bool] = []

    for r in records:
        bid = r["battle_id"]
        si  = r.get("search_info", {})
        mvf = si.get("max_visits_frac")
        qr  = si.get("q_range")
        up  = si.get("using_policy")
        n_leg = si.get("n_actions_legal", 0)
        outcome = battle_outcome.get(bid, 0)
        t  = r["turn"] - battle_min_turn[bid] + 1
        ph = phase(t)

        if mvf is not None:
            all_mvf.append(float(mvf))
            by_phase[ph]["mvf"].append(float(mvf))
            if outcome == 1:
                win_mvf.append(float(mvf))
            elif outcome == -1:
                loss_mvf.append(float(mvf))

        if qr is not None:
            all_qr.append(float(qr))
            by_phase[ph]["qr"].append(float(qr))
            if outcome == 1:
                win_qr.append(float(qr))
            elif outcome == -1:
                loss_qr.append(float(qr))

        if up is not None:
            policy_flags.append(bool(up))

        # Hard-state bucket
        if mvf is not None and qr is not None and n_leg > 0:
            qr_key  = "easy" if float(qr) >= 0.4 else "hard"
            br_key  = "high" if n_leg > 4 else "low"
            hard_buckets[f"{qr_key}+{br_key}"].append(float(mvf))

    policy_pct = sum(policy_flags) / len(policy_flags) if policy_flags else 0.0

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Battles : {n_battles}  ({wins}W / {losses}L / {ties}T)")
    print(f"  Win rate: {win_rate:.1%}")
    print(f"  Avg game length (turns): {avg_len:.1f}")
    print(f"  Using policy: {policy_pct:.1%} of turns")

    # --- Search efficiency by phase ---
    print()
    print(f"  Search efficiency (all turns, n={len(all_mvf)}):")
    print(f"    max_visits_frac : {avg(all_mvf):.3f}  (lower = more distributed)")
    print(f"    q_range         : {avg(all_qr):.3f}  (higher = more decisive)")
    print()
    for ph in ["early", "mid", "late"]:
        mvf_ph = by_phase[ph]["mvf"]
        qr_ph  = by_phase[ph]["qr"]
        if not mvf_ph:
            continue
        print(f"  {ph:6s} (n={len(mvf_ph):4d}):  "
              f"max_visits_frac={avg(mvf_ph):.3f}  "
              f"q_range={avg(qr_ph):.3f}")

    # --- Win/loss search profile ---
    print()
    print(f"  Win / loss search profile:")
    print(f"    wins   (n={len(win_mvf):4d}):  "
          f"max_visits_frac={avg(win_mvf):.3f}  q_range={avg(win_qr):.3f}")
    print(f"    losses (n={len(loss_mvf):4d}):  "
          f"max_visits_frac={avg(loss_mvf):.3f}  q_range={avg(loss_qr):.3f}")

    # --- Hard-state analysis ---
    print()
    print(f"  Hard-state analysis  (max_visits_frac per bucket):")
    print(f"    {'bucket':<20}  {'n':>5}  {'avg_mvf':>8}")
    for key in ["easy+low", "easy+high", "hard+low", "hard+high"]:
        vals = hard_buckets.get(key, [])
        label_str = key.replace("+", "  branch=")
        print(f"    qr={label_str:<18}  {len(vals):>5}  {avg(vals):>8.3f}")
    print(f"    (easy = q_range≥0.4, hard = q_range<0.4; "
          f"low = ≤4 actions, high = >4 actions)")


def compare(path_a: str, label_a: str, path_b: str, label_b: str) -> None:
    recs_a = load(path_a)
    recs_b = load(path_b)
    print(f"\nLoaded {len(recs_a)} turns from {path_a}")
    print(f"Loaded {len(recs_b)} turns from {path_b}")
    summarize(recs_a, label_a)
    summarize(recs_b, label_b)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("file_a", help="Baseline JSONL (e.g. data/ab_heur.jsonl)")
    p.add_argument("file_b", help="Learned JSONL (e.g. data/ab_learned.jsonl)")
    p.add_argument("--label-a", default="Baseline (heuristic prior)")
    p.add_argument("--label-b", default="Learned prior (mixed)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare(args.file_a, args.label_a, args.file_b, args.label_b)
