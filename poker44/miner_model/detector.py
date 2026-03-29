"""
BotDetector: loads a trained sklearn model and scores chunks.

Falls back to a calibrated statistical heuristic if no model file is found,
so the miner can run before training completes.

Model contract:
  - Input:  (N, 60) chunk feature matrix  (from features.extract_chunk_features)
  - Output: probability in [0, 1] that the chunk is bot

Scoring strategy (matches the reward function):
  reward = (0.65 * AP + 0.35 * recall) * max(0, 1 - FPR)^2, zero if FPR >= 0.10
  → Optimise for accurate probability calibration (AP) while keeping FPR < 0.10.
  → Use a slight upward bias on the decision threshold to protect humans.

Model versioning:
  Set MODEL_VERSION env var to load a specific model from models/<version>/model.pkl.
  If unset, falls back to the default model.pkl in this directory.

  Examples:
    MODEL_VERSION=v1_rf_synthetic  → uses the baseline control model
    MODEL_VERSION=v2_rf_mixed      → uses the Phase-1 anti-shortcut-trained model
    (unset)                        → uses model.pkl (legacy default)
"""

from __future__ import annotations

import os
import pickle
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import bittensor as bt

from poker44.miner_model.features import extract_chunk_features

_MODEL_DIR = Path(__file__).parent
_DEFAULT_MODEL_PATH = _MODEL_DIR / "model.pkl"

# Threshold: slightly above 0.5 to reduce false positives on human chunks.
# Tuned to keep FPR < 0.05 empirically.
_DECISION_THRESHOLD = 0.52


def _resolve_model_path() -> Path:
    """
    Resolve which model file to load.

    Priority:
      1. MODEL_VERSION env var → models/<version>/model.pkl
      2. Default model.pkl in this directory
    """
    version = os.environ.get("MODEL_VERSION", "").strip()
    if version and version.lower() not in ("", "default", "latest"):
        versioned = _MODEL_DIR / "models" / version / "model.pkl"
        if versioned.exists():
            return versioned
        # Version dir exists but model not yet trained — warn and fall through
        version_dir = _MODEL_DIR / "models" / version
        if version_dir.exists():
            bt.logging.warning(
                f"[BotDetector] MODEL_VERSION={version!r} directory exists but "
                f"model.pkl not found. Run train.py --version {version} first. "
                f"Falling back to default model."
            )
        else:
            bt.logging.warning(
                f"[BotDetector] MODEL_VERSION={version!r} — no such version directory "
                f"({version_dir}). Falling back to default model."
            )
    return _DEFAULT_MODEL_PATH


class BotDetector:
    """
    Wraps a trained sklearn classifier for chunk-level bot detection.
    Falls back to a hand-crafted heuristic if no model is available.

    The model loaded is determined by MODEL_VERSION env var (see module docstring).
    """

    def __init__(self, model_path: Optional[Path] = None):
        self._model = None
        self._model_path = model_path if model_path is not None else _resolve_model_path()
        self._model_version = os.environ.get("MODEL_VERSION", "default").strip() or "default"
        self._load_model()

    def _load_model(self) -> None:
        if self._model_path.exists():
            bt.logging.info(
                f"[BotDetector] Loading model version={self._model_version!r} "
                f"from {self._model_path} ({self._model_path.stat().st_size:,} bytes)"
            )
            try:
                with open(self._model_path, "rb") as f:
                    self._model = pickle.load(f)
                bt.logging.info(
                    f"[BotDetector] Model loaded: {type(self._model).__name__} "
                    f"(version={self._model_version!r})"
                )
            except Exception as exc:
                bt.logging.error(
                    f"[BotDetector] Failed to load model from {self._model_path}: {exc}\n"
                    f"{traceback.format_exc()}"
                )
                self._model = None
        else:
            bt.logging.warning(
                f"[BotDetector] No model at {self._model_path} "
                f"(version={self._model_version!r}). Using heuristic fallback. "
                f"Run: python -m poker44.miner_model.train --version {self._model_version}"
            )

    def is_model_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_label(self) -> str:
        """Short label for logs: 'ML(v2_rf_mixed)' or 'heuristic'."""
        if self._model is not None:
            return f"ML({self._model_version})"
        return "heuristic"

    def score_chunk(self, chunk: List[Dict[str, Any]]) -> float:
        """Return bot-risk score in [0, 1]. ≥ threshold → predicted bot."""
        if not chunk:
            return 0.5
        if self._model is not None:
            return self._score_with_model(chunk)
        return self._score_heuristic(chunk)

    def predict_chunk(self, chunk: List[Dict[str, Any]]) -> bool:
        return self.score_chunk(chunk) >= _DECISION_THRESHOLD

    # ------------------------------------------------------------------
    # ML path
    # ------------------------------------------------------------------

    def _score_with_model(self, chunk: List[Dict[str, Any]]) -> float:
        feats = extract_chunk_features(chunk).reshape(1, -1)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prob = float(self._model.predict_proba(feats)[0, 1])
        except AttributeError:
            prob = float(self._model.predict(feats)[0])
        return float(np.clip(prob, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Heuristic fallback (used before training, and as a sanity baseline)
    #
    # Key insight: bots are consistent → low within-chunk variance of depth.
    # Humans are diverse → high within-chunk variance of depth.
    # ------------------------------------------------------------------

    def _score_heuristic(self, chunk: List[Dict[str, Any]]) -> float:
        from collections import Counter
        import math

        depths: List[float] = []
        postflop_fracs: List[float] = []
        pots: List[float] = []

        for hand in chunk:
            actions = hand.get("actions") or []
            outcome = hand.get("outcome") or {}

            total_slots = max(len(actions), 1)
            street_counts = Counter(
                str(a.get("street", "")) for a in actions
            )
            postflop_actions = (
                street_counts.get("flop", 0)
                + street_counts.get("turn", 0)
                + street_counts.get("river", 0)
            )
            distinct_postflop = sum(
                1 for s in ("flop", "turn", "river") if street_counts.get(s, 0) > 0
            )
            depths.append(distinct_postflop / 3.0)
            postflop_fracs.append(postflop_actions / total_slots)
            pots.append(float(outcome.get("total_pot") or 0.0))

        n = len(depths)
        if n == 0:
            return 0.5

        mean_depth    = float(np.mean(depths))
        std_depth     = float(np.std(depths))
        mean_postflop = float(np.mean(postflop_fracs))
        std_postflop  = float(np.std(postflop_fracs))
        pot_cv = (float(np.std(pots)) / max(float(np.mean(pots)), 1e-6))

        depth_signal         = mean_depth
        consistency_signal   = max(0.0, 1.0 - std_depth * 4.0)
        postflop_consistency = max(0.0, 1.0 - std_postflop * 3.0)
        pot_consistency      = max(0.0, 1.0 - pot_cv * 0.5)

        score = (
            0.30 * depth_signal
            + 0.35 * consistency_signal
            + 0.20 * postflop_consistency
            + 0.15 * pot_consistency
        )
        return float(np.clip(score, 0.0, 1.0))
