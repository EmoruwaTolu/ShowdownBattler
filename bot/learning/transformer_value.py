"""
Transformer-based value model for the battle state encoder.

Architecture
------------
Input : flat state vector (N_TOTAL = 312,) from encode_state_flat()
        Internally parsed as:
          tokens     (12, 24)  — per-Pokemon tokens
          field      (21,)     — battlefield summary
          active_pair (3,)     — active-vs-active signals

1. Token projection  : (12, 24) → (12, d_model)
2. Side embedding    : add learned [my_team / opp_team] embedding
3. 2× TransformerEncoderLayer (MHA + residual + LayerNorm + FFN + residual + LayerNorm)
4. Mean pool         : (12, d_model) → (d_model,)
5. Context concat    : [pooled ‖ field ‖ active_pair] → (d_model + 24,)
6. Output MLP        : (d_model+24,) → d_out_hidden → 1 → sigmoid

All operations are pure NumPy. Backprop is implemented analytically so the
model can be trained with Adam without any framework dependency.

Interfaces
----------
  predict_prob(tokens, field, active_pair) -> float  win prob [0,1]
  predict_value(tokens, field, active_pair) -> float  MCTS value [-1,1]
  predict_value_from_state(state) -> float            convenience wrapper
  save(path)  /  load(path)                           .npz persistence
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bot.learning.state_encoder import (
    encode_state,
    N_MON_FEATURES,
    N_FIELD_FEATURES,
    N_ACTIVE_PAIR_FEATURES,
    N_TOTAL,
)

# ------------------------------------------------------------------ constants
_SEQ   = 12       # number of Pokemon slots
_SIDE  = [0]*6 + [1]*6   # side id per slot (my=0, opp=1)


# ------------------------------------------------------------------ pure math
def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


# ------------------------------------------------------------------ layer norm
def _ln_fwd(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5,
) -> Tuple[np.ndarray, tuple]:
    """Layer norm forward.  x: (..., d)"""
    mu    = x.mean(axis=-1, keepdims=True)
    var   = x.var(axis=-1, keepdims=True)
    sigma = np.sqrt(var + eps)
    x_hat = (x - mu) / sigma
    return gamma * x_hat + beta, (x_hat, sigma, gamma)


def _ln_bwd(
    d_out: np.ndarray,
    cache: tuple,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (dx, d_gamma, d_beta)."""
    x_hat, sigma, gamma = cache
    d = x_hat.shape[-1]
    dx_hat   = d_out * gamma
    dx       = (dx_hat
                - dx_hat.mean(axis=-1, keepdims=True)
                - x_hat * (dx_hat * x_hat).mean(axis=-1, keepdims=True)
                ) / sigma
    # Sum over every axis except the last (feature axis).
    reduce   = tuple(range(len(d_out.shape) - 1))
    d_gamma  = (d_out * x_hat).sum(axis=reduce)
    d_beta   = d_out.sum(axis=reduce)
    return dx, d_gamma, d_beta


# --------------------------------------------------------- multi-head attention
def _mha_fwd(
    x: np.ndarray,
    Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray, Wo: np.ndarray,
    n_heads: int,
) -> Tuple[np.ndarray, tuple]:
    """
    x   : (B, seq, d_model)
    W*  : (d_model, d_model)
    out : (B, seq, d_model)
    """
    B, seq, d = x.shape
    dh = d // n_heads

    Q_flat = x @ Wq          # (B, seq, d)
    K_flat = x @ Wk
    V_flat = x @ Wv

    # reshape → (B, n_heads, seq, dh)
    def _split(z):
        return z.reshape(B, seq, n_heads, dh).transpose(0, 2, 1, 3)

    Q, K, V = _split(Q_flat), _split(K_flat), _split(V_flat)

    scale  = math.sqrt(dh)
    scores = (Q @ K.transpose(0, 1, 3, 2)) / scale   # (B, H, seq, seq)
    attn   = _softmax(scores, axis=-1)

    out_h  = (attn @ V).transpose(0, 2, 1, 3).reshape(B, seq, d)  # (B, seq, d)
    out    = out_h @ Wo

    cache = (x, Q_flat, K_flat, V_flat, Q, K, V, attn, out_h, Wq, Wk, Wv, Wo, n_heads, scale)
    return out, cache


def _mha_bwd(
    d_out: np.ndarray,
    cache: tuple,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (dx, dWq, dWk, dWv, dWo)."""
    x, Q_flat, K_flat, V_flat, Q, K, V, attn, out_h, Wq, Wk, Wv, Wo, n_heads, scale = cache
    B, seq, d = x.shape
    dh = d // n_heads

    # ── output projection
    Bs = B * seq
    dWo    = out_h.reshape(Bs, d).T @ d_out.reshape(Bs, d)
    d_out_h = d_out @ Wo.T                              # (B, seq, d)

    # reshape to (B, n_heads, seq, dh)
    d_attn_V = d_out_h.reshape(B, seq, n_heads, dh).transpose(0, 2, 1, 3)

    # ── attn @ V
    d_attn = d_attn_V @ V.transpose(0, 1, 3, 2)         # (B, H, seq, seq)
    dV     = attn.transpose(0, 1, 3, 2) @ d_attn_V      # (B, H, seq, dh)

    # ── softmax backward: s*(d - sum(d*s))
    d_scores = attn * (d_attn - (d_attn * attn).sum(axis=-1, keepdims=True))
    d_scores /= scale

    # ── Q @ K.T
    dQ = d_scores @ K                                    # (B, H, seq, dh)
    dK = d_scores.transpose(0, 1, 3, 2) @ Q

    # reshape back to (B, seq, d)
    def _merge(z):
        return z.transpose(0, 2, 1, 3).reshape(B, seq, d)

    dQ_flat, dK_flat, dV_flat = _merge(dQ), _merge(dK), _merge(dV)

    xf = x.reshape(Bs, d)
    dWq = xf.T @ dQ_flat.reshape(Bs, d)
    dWk = xf.T @ dK_flat.reshape(Bs, d)
    dWv = xf.T @ dV_flat.reshape(Bs, d)

    dx = dQ_flat @ Wq.T + dK_flat @ Wk.T + dV_flat @ Wv.T
    return dx, dWq, dWk, dWv, dWo


# ------------------------------------------------------ transformer encoder layer
def _enc_fwd(x: np.ndarray, p: Dict) -> Tuple[np.ndarray, tuple]:
    """
    One encoder layer: MHA sublayer + FFN sublayer, both with residual + LayerNorm.
    p : parameter dict for this layer.
    """
    # ── MHA sublayer
    mha_out, mha_cache = _mha_fwd(x, p["Wq"], p["Wk"], p["Wv"], p["Wo"], p["n_heads"])
    x1               = x + mha_out
    x1_ln, ln1_cache = _ln_fwd(x1, p["ln1_g"], p["ln1_b"])

    # ── FFN sublayer
    ffn_pre  = x1_ln @ p["W1"] + p["b1"]              # (B, seq, d_ff)
    ffn_h    = _relu(ffn_pre)
    ffn_out  = ffn_h @ p["W2"] + p["b2"]              # (B, seq, d)
    x2       = x1_ln + ffn_out
    x2_ln, ln2_cache = _ln_fwd(x2, p["ln2_g"], p["ln2_b"])

    cache = (x, mha_cache, x1, ln1_cache, x1_ln, ffn_pre, ffn_h, ffn_out, x2, ln2_cache, p)
    return x2_ln, cache


def _enc_bwd(d_out: np.ndarray, cache: tuple) -> Tuple[np.ndarray, Dict]:
    """Returns (dx, grad_dict matching parameter dict keys)."""
    x, mha_cache, x1, ln1_cache, x1_ln, ffn_pre, ffn_h, ffn_out, x2, ln2_cache, p = cache
    B, seq, d = x.shape
    d_ff = p["W1"].shape[-1]

    # ── LN2 + residual
    d_x2, d_ln2_g, d_ln2_b = _ln_bwd(d_out, ln2_cache)
    d_x1_ln  = d_x2.copy()   # residual: grad flows to x1_ln directly
    d_ffn_out = d_x2

    # ── FFN backward
    Bs = B * seq
    d_ffn_h      = d_ffn_out @ p["W2"].T                  # (B, seq, d_ff)
    d_ffn_pre    = d_ffn_h * (ffn_pre > 0)                # relu
    dW2          = ffn_h.reshape(Bs, d_ff).T @ d_ffn_out.reshape(Bs, d)
    db2          = d_ffn_out.sum(axis=(0, 1))
    dW1          = x1_ln.reshape(Bs, d).T @ d_ffn_pre.reshape(Bs, d_ff)
    db1          = d_ffn_pre.sum(axis=(0, 1))
    d_x1_ln     += d_ffn_pre @ p["W1"].T

    # ── LN1 + residual
    d_x1, d_ln1_g, d_ln1_b = _ln_bwd(d_x1_ln, ln1_cache)
    d_mha_out = d_x1
    d_x_res   = d_x1   # residual from x + mha_out

    # ── MHA backward
    d_x_mha, dWq, dWk, dWv, dWo = _mha_bwd(d_mha_out, mha_cache)
    dx = d_x_res + d_x_mha

    grads = dict(Wq=dWq, Wk=dWk, Wv=dWv, Wo=dWo,
                 W1=dW1, b1=db1, W2=dW2, b2=db2,
                 ln1_g=d_ln1_g, ln1_b=d_ln1_b,
                 ln2_g=d_ln2_g, ln2_b=d_ln2_b)
    return dx, grads


# ================================================================ model class
class TransformerValueModel:
    """
    Transformer value model.  All weights are float64 numpy arrays.

    Typical usage
    -------------
    # Inference from a ShadowState:
    value = model.predict_value_from_state(state)

    # Training (see train_transformer.py):
    loss, grads = model.forward_backward(flat_batch, y_batch, weights)
    model.adam_step(grads, lr=1e-3, l2=1e-4)
    """

    def __init__(
        self,
        d_model: int     = 64,
        n_heads: int     = 4,
        n_layers: int    = 2,
        d_ff: int        = 128,
        d_out_hidden: int = 32,
        seed: int        = 42,
    ):
        self.d_model      = d_model
        self.n_heads      = n_heads
        self.n_layers     = n_layers
        self.d_ff         = d_ff
        self.d_out_hidden = d_out_hidden

        rng = np.random.default_rng(seed)
        ctx_dim = d_model + N_FIELD_FEATURES + N_ACTIVE_PAIR_FEATURES  # 88

        # ── token projection
        self.W_tok = rng.normal(0, math.sqrt(2.0 / N_MON_FEATURES), (N_MON_FEATURES, d_model))
        self.b_tok = np.zeros(d_model)

        # ── side embedding  (slot 0-5 = my team, 6-11 = opp)
        self.side_emb = rng.normal(0, 0.02, (2, d_model))

        # ── encoder layers
        self.layers: List[Dict] = []
        for _ in range(n_layers):
            std_w = math.sqrt(2.0 / d_model)
            self.layers.append({
                "Wq":    rng.normal(0, std_w, (d_model, d_model)),
                "Wk":    rng.normal(0, std_w, (d_model, d_model)),
                "Wv":    rng.normal(0, std_w, (d_model, d_model)),
                "Wo":    rng.normal(0, std_w, (d_model, d_model)),
                "W1":    rng.normal(0, math.sqrt(2.0 / d_model), (d_model, d_ff)),
                "b1":    np.zeros(d_ff),
                "W2":    rng.normal(0, math.sqrt(2.0 / d_ff), (d_ff, d_model)),
                "b2":    np.zeros(d_model),
                "ln1_g": np.ones(d_model),
                "ln1_b": np.zeros(d_model),
                "ln2_g": np.ones(d_model),
                "ln2_b": np.zeros(d_model),
                "n_heads": n_heads,
            })

        # ── output head
        self.W_h   = rng.normal(0, math.sqrt(2.0 / ctx_dim), (ctx_dim, d_out_hidden))
        self.b_h   = np.zeros(d_out_hidden)
        self.w_out = np.zeros(d_out_hidden)
        self.b_out = 0.0

        # Adam state  (initialised lazily in adam_step)
        self._adam_m: Optional[Dict] = None
        self._adam_v: Optional[Dict] = None
        self._adam_t: int = 0

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _parse(flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split (B, N_TOTAL) → tokens (B,12,24), field (B,21), ap (B,3)."""
        B = flat.shape[0]
        tokens = flat[:, :_SEQ * N_MON_FEATURES].reshape(B, _SEQ, N_MON_FEATURES)
        field  = flat[:, _SEQ * N_MON_FEATURES:_SEQ * N_MON_FEATURES + N_FIELD_FEATURES]
        ap     = flat[:, _SEQ * N_MON_FEATURES + N_FIELD_FEATURES:]
        return tokens, field, ap

    # ------------------------------------------------------------ forward pass
    def _forward(
        self,
        flat: np.ndarray,
        return_cache: bool = False,
    ) -> Tuple[np.ndarray, Optional[tuple]]:
        """
        flat : (B, N_TOTAL)
        Returns (probs (B,), cache) where cache is None if return_cache=False.
        """
        tokens, field, ap = self._parse(flat)
        B = flat.shape[0]

        # 1. Token projection
        x = tokens @ self.W_tok + self.b_tok  # (B, 12, d_model)

        # 2. Side embedding
        side_ids = np.array(_SIDE, dtype=np.int32)         # (12,)
        emb = self.side_emb[side_ids]                       # (12, d_model)
        x = x + emb                                         # broadcast over B

        # 3. Encoder layers
        enc_caches = []
        for layer in self.layers:
            x, ec = _enc_fwd(x, layer)
            enc_caches.append(ec)

        # 4. Mean pool
        pooled = x.mean(axis=1)                             # (B, d_model)

        # 5. Context concat
        ctx = np.concatenate([pooled, field, ap], axis=1)  # (B, 88)

        # 6. Output MLP
        h      = _relu(ctx @ self.W_h + self.b_h)          # (B, d_out_hidden)
        logit  = h @ self.w_out + self.b_out               # (B,)
        probs  = _sigmoid(logit)

        cache = (tokens, field, ap, x, enc_caches, pooled, ctx, h, logit, probs) \
                if return_cache else None
        return probs, cache

    # ----------------------------------------------- forward + backward pass
    def forward_backward(
        self,
        flat: np.ndarray,
        y: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict]:
        """
        flat           : (B, N_TOTAL) float64
        y              : (B,)  binary labels (1=win, 0=loss)
        sample_weights : (B,)  per-sample weights (None = uniform)

        Returns (weighted_bce_loss, grad_dict).
        grad_dict keys: "W_tok", "b_tok", "side_emb",
                        "layer_0_Wq", "layer_0_Wk", ..., "layer_1_...",
                        "W_h", "b_h", "w_out", "b_out"
        """
        B = flat.shape[0]
        sw = sample_weights if sample_weights is not None else np.ones(B)
        sw = sw / (sw.mean() + 1e-12)

        # ── forward
        probs, cache = self._forward(flat, return_cache=True)
        tokens, field, ap, x_out, enc_caches, pooled, ctx, h, logit, _ = cache

        # ── BCE loss
        p_clip = np.clip(probs, 1e-12, 1.0 - 1e-12)
        per_sample = -(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip))
        loss = float((sw * per_sample).mean())

        # ── backward: loss → sigmoid
        d_logit = sw * (probs - y) / B                     # (B,)

        # ── output MLP backward
        d_w_out = h.T @ d_logit                            # (d_out_hidden,)
        d_b_out = float(d_logit.sum())
        d_h     = np.outer(d_logit, self.w_out)            # (B, d_out_hidden)
        d_h_pre = d_h * (h > 0)                            # relu
        d_W_h   = ctx.T @ d_h_pre                          # (88, d_out_hidden)
        d_b_h   = d_h_pre.sum(axis=0)
        d_ctx   = d_h_pre @ self.W_h.T                     # (B, 88)

        # ── split context gradient
        d_pooled = d_ctx[:, :self.d_model]
        # (field and ap gradients are not propagated — they are fixed encodings)

        # ── mean pool backward: distribute equally to all 12 positions
        d_x = np.tile(d_pooled[:, np.newaxis, :], (1, _SEQ, 1)) / _SEQ   # (B,12,d)

        # ── encoder layers backward (in reverse)
        layer_grads: List[Dict] = [None] * self.n_layers  # type: ignore
        for i in reversed(range(self.n_layers)):
            d_x, lg = _enc_bwd(d_x, enc_caches[i])
            layer_grads[i] = lg

        # ── side embedding backward
        d_side_emb = np.zeros_like(self.side_emb)
        for si, side_id in enumerate(_SIDE):
            d_side_emb[side_id] += d_x[:, si, :].sum(axis=0)

        # ── token projection backward
        B_seq = B * _SEQ
        tokens_flat = self._parse(flat)[0].reshape(B_seq, N_MON_FEATURES)
        d_x_flat    = d_x.reshape(B_seq, self.d_model)
        d_W_tok = tokens_flat.T @ d_x_flat                 # (N_MON_FEATURES, d_model)
        d_b_tok = d_x_flat.sum(axis=0)

        # ── collect grads
        grads: Dict = {
            "W_tok":    d_W_tok,
            "b_tok":    d_b_tok,
            "side_emb": d_side_emb,
            "W_h":      d_W_h,
            "b_h":      d_b_h,
            "w_out":    d_w_out,
            "b_out":    d_b_out,
        }
        for i, lg in enumerate(layer_grads):
            for k, v in lg.items():
                grads[f"layer_{i}_{k}"] = v

        return loss, grads

    # ---------------------------------------------------- Adam optimizer step
    def adam_step(
        self,
        grads: Dict,
        lr: float = 1e-3,
        l2: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        if self._adam_m is None:
            self._adam_m = {k: np.zeros_like(v) if isinstance(v, np.ndarray) else 0.0
                            for k, v in grads.items()}
            self._adam_v = {k: np.zeros_like(v) if isinstance(v, np.ndarray) else 0.0
                            for k, v in grads.items()}
        self._adam_t += 1
        t = self._adam_t
        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2 ** t

        def _update(param, grad, m, v):
            if isinstance(grad, float):
                g   = grad + l2 * param
                m_  = beta1 * m + (1 - beta1) * g
                v_  = beta2 * v + (1 - beta2) * g ** 2
                step = lr * (m_ / bc1) / (math.sqrt(v_ / bc2) + eps)
                return param - step, m_, v_
            else:
                g   = grad + l2 * param
                m_  = beta1 * m + (1 - beta1) * g
                v_  = beta2 * v + (1 - beta2) * g ** 2
                step = lr * (m_ / bc1) / (np.sqrt(v_ / bc2) + eps)
                return param - step, m_, v_

        def _apply(name: str, param):
            if name not in grads:
                return param
            p, m, v = _update(param, grads[name], self._adam_m[name], self._adam_v[name])
            self._adam_m[name] = m
            self._adam_v[name] = v
            return p

        self.W_tok    = _apply("W_tok",    self.W_tok)
        self.b_tok    = _apply("b_tok",    self.b_tok)
        self.side_emb = _apply("side_emb", self.side_emb)
        self.W_h      = _apply("W_h",      self.W_h)
        self.b_h      = _apply("b_h",      self.b_h)
        self.w_out    = _apply("w_out",    self.w_out)
        self.b_out    = _apply("b_out",    self.b_out)
        for i, layer in enumerate(self.layers):
            for k in list(layer.keys()):
                if k == "n_heads":
                    continue
                layer[k] = _apply(f"layer_{i}_{k}", layer[k])

    # ---------------------------------------------------------- inference API
    def predict_prob(
        self,
        tokens: np.ndarray,
        field: np.ndarray,
        active_pair: np.ndarray,
    ) -> float:
        """Win probability [0, 1] from the three encoder arrays."""
        flat = np.concatenate([
            tokens.reshape(-1),
            field,
            active_pair,
        ], dtype=np.float64)[np.newaxis, :]       # (1, 312)
        probs, _ = self._forward(flat, return_cache=False)
        return float(probs[0])

    def predict_value(
        self,
        tokens: np.ndarray,
        field: np.ndarray,
        active_pair: np.ndarray,
    ) -> float:
        """MCTS-compatible value in [-1, 1]  (win_prob → 2p - 1)."""
        return 2.0 * self.predict_prob(tokens, field, active_pair) - 1.0

    def predict_value_from_state(self, state: Any) -> float:
        """
        Convenience wrapper: encode a ShadowState and return the MCTS value.
        Drop-in replacement for ValueModel.predict_value(eval_terms).
        """
        try:
            tokens, field, ap = encode_state(state)
            return self.predict_value(
                tokens.astype(np.float64),
                field.astype(np.float64),
                ap.astype(np.float64),
            )
        except Exception:
            return 0.0

    # ---------------------------------------------------------- save / load
    def save(self, path: str) -> None:
        """Save all weights to a .npz file."""
        d: Dict = {
            "d_model":       np.array([self.d_model]),
            "n_heads":       np.array([self.n_heads]),
            "n_layers":      np.array([self.n_layers]),
            "d_ff":          np.array([self.d_ff]),
            "d_out_hidden":  np.array([self.d_out_hidden]),
            "W_tok":         self.W_tok,
            "b_tok":         self.b_tok,
            "side_emb":      self.side_emb,
            "W_h":           self.W_h,
            "b_h":           self.b_h,
            "w_out":         self.w_out,
            "b_out":         np.array([self.b_out]),
        }
        for i, layer in enumerate(self.layers):
            for k, v in layer.items():
                if k != "n_heads":
                    d[f"layer_{i}_{k}"] = v
        np.savez(path, **d)

    @classmethod
    def load(cls, path: str) -> "TransformerValueModel":
        """Load weights from a .npz file saved by save()."""
        data = np.load(path, allow_pickle=False)
        model = cls(
            d_model      = int(data["d_model"][0]),
            n_heads      = int(data["n_heads"][0]),
            n_layers     = int(data["n_layers"][0]),
            d_ff         = int(data["d_ff"][0]),
            d_out_hidden = int(data["d_out_hidden"][0]),
        )
        model.W_tok    = data["W_tok"].astype(np.float64)
        model.b_tok    = data["b_tok"].astype(np.float64)
        model.side_emb = data["side_emb"].astype(np.float64)
        model.W_h      = data["W_h"].astype(np.float64)
        model.b_h      = data["b_h"].astype(np.float64)
        model.w_out    = data["w_out"].astype(np.float64)
        model.b_out    = float(data["b_out"][0])
        for i, layer in enumerate(model.layers):
            for k in list(layer.keys()):
                if k != "n_heads":
                    key = f"layer_{i}_{k}"
                    if key in data:
                        layer[k] = data[key].astype(np.float64)
        return model
