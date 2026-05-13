"""
Sequence-based bot detector (v10+).

Motivation
----------
Top-performing miners on the Poker44 dashboard now use sequence / tensor
architectures (e.g. `poker44_ml13tens1`). Our gradient-boosting stack appears
to be capped around composite ≈ 0.40, while the tensor models reach ≈ 0.60.

The hypothesis: a chunk is fundamentally a *sequence* of ~60 hands belonging
to the same player. Mean/std/p25/p75 aggregates discard the temporal signal
("is the hero playing consistently across the session?"). A small Bi-LSTM
over the per-hand feature vectors keeps that signal.

Architecture
------------
    Input   : (B, T, F)  per-hand features, F = _N_HAND_FEATURES (43)
    Norm    : LayerNorm over feature dim
    Encoder : 1-layer Bi-LSTM, hidden = 64  ->  (B, T, 128)
    Pool    : mask-aware additive attention -> (B, 128)
    Head    : Linear 128 -> 64 -> 1  (BCE-with-logits)

~63k parameters; runs in well under 100 ms per 40-chunk batch on CPU with one
thread, so it fits inside the validator timeout with plenty of headroom.

The wrapper class implements `predict_proba(chunks)` so the existing
`BotDetector.score_chunks_batch` path can drive it with one extra branch
(detector.py knows it's a sequence model via `_is_sequence_model = True`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:  # pragma: no cover - dev-only guard
    _TORCH_OK = False

from poker44.miner_model.features import _N_HAND_FEATURES, extract_hand_matrix


# ----------------------------------------------------------------------
# Torch model
# ----------------------------------------------------------------------

if _TORCH_OK:

    class HandSeqClassifier(nn.Module):
        """Bi-LSTM over the per-hand feature sequence + attention pooling."""

        def __init__(
            self,
            n_features: int = _N_HAND_FEATURES,
            hidden: int = 64,
            num_layers: int = 1,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.n_features = n_features
            self.hidden = hidden
            self.num_layers = num_layers

            self.input_norm = nn.LayerNorm(n_features)
            self.encoder = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden,
                num_layers=num_layers,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            enc_out = hidden * 2
            self.attn = nn.Linear(enc_out, 1)
            self.head = nn.Sequential(
                nn.Linear(enc_out, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(
            self,
            x: "torch.Tensor",
            mask: "torch.Tensor",
        ) -> "torch.Tensor":
            """
            Args:
                x:    (B, T, F) per-hand features
                mask: (B, T) bool, True for valid positions

            Returns:
                logits (B,) -- pre-sigmoid bot probability logit
            """
            x = self.input_norm(x)
            out, _ = self.encoder(x)  # (B, T, 2H)
            scores = self.attn(out).squeeze(-1)  # (B, T)
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = F.softmax(scores, dim=1)
            pooled = (out * weights.unsqueeze(-1)).sum(dim=1)  # (B, 2H)
            logits = self.head(pooled).squeeze(-1)  # (B,)
            return logits


def chunks_to_padded_batch(
    chunks: List[List[Dict[str, Any]]],
    *,
    max_hands: int = 120,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of raw chunks into padded arrays ready for the seq model.

    Returns:
        x    : (B, T, F) float32, padded with zeros
        mask : (B, T)    bool,    True for valid hand positions

    Hands beyond ``max_hands`` are truncated; chunks with zero hands keep at
    least one position so the model can produce a valid logit.
    """
    matrices = [extract_hand_matrix(c)[:max_hands] for c in chunks]
    lens = [m.shape[0] for m in matrices]
    T = max(max(lens), 1)
    B = len(matrices)
    F = _N_HAND_FEATURES

    x = np.zeros((B, T, F), dtype=np.float32)
    mask = np.zeros((B, T), dtype=np.bool_)
    for i, m in enumerate(matrices):
        if m.shape[0] == 0:
            mask[i, 0] = True  # avoid all-False softmax
            continue
        n = m.shape[0]
        x[i, :n] = m
        mask[i, :n] = True
    return x, mask


# ----------------------------------------------------------------------
# Pickle-friendly wrapper exposing a sklearn-like API
# ----------------------------------------------------------------------

class SequenceModelWrapper:
    """
    Wraps a trained ``HandSeqClassifier`` so it can be pickled and loaded by
    ``BotDetector`` like any sklearn model. Exposes:

        - predict_proba(chunks: List[chunk]) -> np.ndarray (N, 2)
        - n_features_in_ for warm-up dim detection
        - _is_sequence_model = True so detector.py routes to the chunk path

    State stored on disk is the model config + state_dict (raw tensors), not
    the live ``nn.Module`` object, so it survives torch version bumps.
    """

    _is_sequence_model = True

    def __init__(self, config: Dict[str, Any], state_dict: Dict[str, Any]) -> None:
        self._config = dict(config)
        self._state_dict = state_dict
        self._model: Any = None  # lazy-init torch module
        self.n_features_in_ = int(config.get("n_features", _N_HAND_FEATURES))

    # --- pickle support -----------------------------------------------
    def __getstate__(self) -> Dict[str, Any]:
        # Strip live torch module; keep only config + cpu state_dict
        sd = self._state_dict
        if self._model is not None:
            try:
                sd = {k: v.detach().cpu() for k, v in self._model.state_dict().items()}
            except Exception:
                pass
        return {"_config": self._config, "_state_dict": sd}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self._config = state["_config"]
        self._state_dict = state["_state_dict"]
        self._model = None
        self.n_features_in_ = int(self._config.get("n_features", _N_HAND_FEATURES))

    # --- inference ----------------------------------------------------
    def _ensure_model(self) -> None:
        if not _TORCH_OK:
            raise RuntimeError("torch is required for SequenceModelWrapper")
        if self._model is not None:
            return
        torch.set_num_threads(1)  # match BLAS-thread strategy in detector.py
        m = HandSeqClassifier(
            n_features=self._config["n_features"],
            hidden=self._config["hidden"],
            num_layers=self._config.get("num_layers", 1),
            dropout=0.0,  # eval mode
        )
        m.load_state_dict(self._state_dict)
        m.eval()
        self._model = m

    @torch.inference_mode() if _TORCH_OK else (lambda f: f)
    def predict_proba(
        self,
        chunks_or_array: Any,
    ) -> np.ndarray:
        """
        Returns (N, 2) probabilities like sklearn classifiers.
        Accepts EITHER a list of raw chunks OR a numpy array of pre-extracted
        per-hand matrices. The detector path passes raw chunks.
        """
        self._ensure_model()

        # Normalise input: dummy 2-D ndarrays come from the warm-up pass.
        # Treat them as already-padded (B, T, F) if 3-D, else map to a single
        # zero chunk so warm-up still exercises the forward path.
        if isinstance(chunks_or_array, np.ndarray):
            arr = chunks_or_array
            if arr.ndim == 2:
                # Warm-up dummy: BotDetector creates np.zeros((4, n_features))
                # treat each row as a 1-hand chunk so forward succeeds.
                B = arr.shape[0]
                T = max(1, 1)
                x = np.zeros((B, T, self._config["n_features"]), dtype=np.float32)
                mask = np.ones((B, T), dtype=np.bool_)
            elif arr.ndim == 3:
                x = arr.astype(np.float32, copy=False)
                mask = np.ones(arr.shape[:2], dtype=np.bool_)
            else:
                raise ValueError(f"unsupported ndarray shape {arr.shape}")
        else:
            x, mask = chunks_to_padded_batch(
                chunks_or_array,
                max_hands=int(self._config.get("max_hands", 120)),
            )

        xt = torch.from_numpy(x)
        mt = torch.from_numpy(mask)
        logits = self._model(xt, mt)  # (B,)
        probs1 = torch.sigmoid(logits).cpu().numpy()
        out = np.empty((probs1.shape[0], 2), dtype=np.float32)
        out[:, 0] = 1.0 - probs1
        out[:, 1] = probs1
        return out


__all__ = [
    "HandSeqClassifier",
    "SequenceModelWrapper",
    "chunks_to_padded_batch",
]
