"""
Lightweight policy model wrapper.

Supports both linear and 1-hidden-layer MLP models.

Feature vector per action (50 features, must match train_policy.py):
  [state_vec (24)]        — eval_terms in feature_order
  [action_features (26)]  — per-action features from action_features.py
                            (includes heuristic_score as feature #25)

All features are z-score normalized using feat_mean / feat_std saved at train time.

Model detection: if the .npz file contains "W1", it's an MLP; otherwise linear.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bot.learning.action_features import compute_action_features, ACTION_FEATURE_NAMES

_N_ACTION = len(ACTION_FEATURE_NAMES)  # 25


class PolicyModel:
    def __init__(
        self,
        feature_order: List[str],
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
        # Linear params (hidden_size == 0)
        w: Optional[np.ndarray] = None,
        b: float = 0.0,
        # MLP params (hidden_size > 0)
        W1: Optional[np.ndarray] = None,
        b1: Optional[np.ndarray] = None,
        w2: Optional[np.ndarray] = None,
        b2: float = 0.0,
    ):
        self.feature_order = feature_order
        self.feat_mean = feat_mean
        self.feat_std = feat_std
        self.is_mlp = W1 is not None
        if self.is_mlp:
            self.W1 = W1
            self.b1 = b1
            self.w2 = w2
            self.b2 = b2
        else:
            self.w = w
            self.b = b

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @classmethod
    def load(cls, path: str) -> "PolicyModel":
        data = np.load(path, allow_pickle=False)
        feature_order = list(data["feature_order"].astype(str))
        feat_mean = data["feat_mean"].astype(np.float64)
        feat_std  = data["feat_std"].astype(np.float64)
        if "W1" in data:
            return cls(
                feature_order=feature_order,
                feat_mean=feat_mean,
                feat_std=feat_std,
                W1=data["W1"].astype(np.float64),
                b1=data["b1"].astype(np.float64),
                w2=data["w2"].astype(np.float64),
                b2=float(data["b2"][0]),
            )
        else:
            return cls(
                feature_order=feature_order,
                feat_mean=feat_mean,
                feat_std=feat_std,
                w=data["w"].astype(np.float64),
                b=float(data["b"][0]),
            )

    def _build_feature(
        self,
        eval_terms: Dict[str, float],
        action_feats: Dict[str, float],
    ) -> np.ndarray:
        state_vec = np.array(
            [eval_terms.get(k, 0.0) for k in self.feature_order],
            dtype=np.float64,
        )
        action_vec = np.array(
            [action_feats.get(k, 0.0) for k in ACTION_FEATURE_NAMES],
            dtype=np.float64,
        )
        raw = np.concatenate([state_vec, action_vec])  # (49,)
        return (raw - self.feat_mean) / self.feat_std

    def score_actions(
        self,
        eval_terms: Dict[str, float],
        actions: List[Tuple[str, Any]],
        battle: Any = None,
        ctx_me: Any = None,
        heuristic_scores: Optional[List[float]] = None,
    ) -> List[float]:
        """
        Returns one unnormalized logit per action (same order as `actions`).
        action_priors() softmaxes over these.

        battle + ctx_me: used to compute per-action features at inference.
        heuristic_scores: raw heuristic logits (one per action) — if provided,
            used as the heuristic_score feature. If not provided, defaults to 0.
        """
        scores = []
        for i, (kind, obj) in enumerate(actions):
            if battle is not None and ctx_me is not None:
                try:
                    af = compute_action_features(kind, obj, battle, ctx_me)
                except Exception:
                    af = {}
            else:
                af = {}
            # Inject heuristic_score feature (set externally, not by compute_action_features)
            if heuristic_scores is not None and i < len(heuristic_scores):
                af["heuristic_score"] = float(heuristic_scores[i])
            feat = self._build_feature(eval_terms, af)  # (50,)
            if self.is_mlp:
                h = self._relu(feat @ self.W1 + self.b1)
                score = float(h @ self.w2 + self.b2)
            else:
                score = float(feat @ self.w + self.b)
            scores.append(score)
        return scores
