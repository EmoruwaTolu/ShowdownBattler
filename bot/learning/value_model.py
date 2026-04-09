"""
Lightweight value model wrapper.

Input:  eval_terms dict (24 state features, z-score normalized)
Output: win probability in [0, 1],  or value in [-1, 1] for MCTS leaf eval

Model detection: if the .npz file contains "W1", it's an MLP; otherwise logistic regression.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np


class ValueModel:
    def __init__(
        self,
        feature_order: List[str],
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
        # Logistic regression params
        w: Optional[np.ndarray] = None,
        b: float = 0.0,
        # MLP params
        W1: Optional[np.ndarray] = None,
        b1: Optional[np.ndarray] = None,
        w2: Optional[np.ndarray] = None,
        b2: float = 0.0,
    ):
        self.feature_order = feature_order
        self.feat_mean = feat_mean
        self.feat_std  = feat_std
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

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    @classmethod
    def load(cls, path: str) -> "ValueModel":
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

    def _build_feature(self, eval_terms: Dict[str, float]) -> np.ndarray:
        raw = np.array(
            [eval_terms.get(k, 0.0) for k in self.feature_order],
            dtype=np.float64,
        )
        return (raw - self.feat_mean) / self.feat_std

    def predict_prob(self, eval_terms: Dict[str, float]) -> float:
        """Win probability in [0, 1]."""
        feat = self._build_feature(eval_terms)
        if self.is_mlp:
            h = self._relu(feat @ self.W1 + self.b1)
            logit = float(h @ self.w2 + self.b2)
        else:
            logit = float(feat @ self.w + self.b)
        return self._sigmoid(logit)

    def predict_value(self, eval_terms: Dict[str, float]) -> float:
        """
        Value in [-1, 1] compatible with evaluate_state() output.
        Maps win_prob=1.0 → +1.0, win_prob=0.0 → -1.0.
        """
        return 2.0 * self.predict_prob(eval_terms) - 1.0
