"""
BotDetector: loads a trained sklearn model and scores chunks.

Falls back to a calibrated statistical heuristic if no model file is found,
so the miner can run before training completes.

Model contract:
  - Input:  (N, 76) chunk feature matrix  (from features.extract_chunk_features)
  - Output: probability in [0, 1] that the chunk is bot

Scoring strategy (matches the reward function):
  reward = (0.65 * AP + 0.35 * recall) * max(0, 1 - FPR)^2, zero if FPR >= 0.10
  → Optimise for accurate probability calibration (AP) while keeping FPR < 0.10.

  IMPORTANT: the validator thresholds your raw risk_scores at 0.5 (np.round).
  Borderline scores of 0.51–0.55 become false positives and spike FPR.
  A conservative bias shifts uncertain predictions safely below 0.5.

  Score bias applied: raw_prob - SCORE_BIAS (default 0.06).
  Effect: a score of 0.55 → 0.49 (human, safe). A score of 0.80 → 0.74 (still bot).
  Trade-off: negligible recall loss on confident predictions, significant FPR protection
  on borderline cases.

Model versioning:
  Set MODEL_VERSION env var to load a specific model from models/<version>/model.pkl.
  If unset, falls back to the default model.pkl in this directory.

  Examples:
    MODEL_VERSION=v1_rf_synthetic  → baseline control model (60 features — retrain needed)
    MODEL_VERSION=v2_rf_mixed      → Phase-1 anti-shortcut model (60 features — retrain needed)
    MODEL_VERSION=v3_gb_mixed      → Phase-2 HistGBM model, 76 features (recommended)
    (unset)                        → model.pkl (legacy default)

  NOTE: if you load a model trained with 60 features but the code now extracts 76
  features, BotDetector will catch the dimension mismatch and fall back to the heuristic.
  Retrain with --version v3_gb_mixed to use the full feature set.
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

# Conservative score bias: shifts all raw model scores down by this amount
# before returning. This pushes borderline human chunks (0.50–0.56) below the
# validator's 0.5 rounding threshold, protecting against FPR spikes.
# It has negligible effect on confident bot predictions (0.75+ stays bot).
# Bias was used to compensate for isotonic calibration pushing scores just
# above 0.5 for humans. With the new heuristic and v7 model this is no
# longer needed and would systematically downshift graded probabilities.
_SCORE_BIAS = 0.0


def _limit_threads():
    """Limit BLAS/OpenMP threads to 1 for small inference matrices.

    On high-core-count CPUs (16+), numpy/sklearn spin up all available
    threads even for tiny (40×100) matrices. The thread synchronization
    overhead completely dominates the actual computation, causing 60-second
    delays instead of <1-second ones. Limiting to 1 thread for inference
    eliminates this overhead while having zero impact on accuracy.
    """
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1)
    except ImportError:
        import contextlib
        return contextlib.nullcontext()


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
    Falls back to a hand-crafted heuristic if no model is available or if
    a feature-dimension mismatch is detected (e.g. old model, new features).

    The model loaded is determined by MODEL_VERSION env var (see module docstring).
    """

    def __init__(self, model_path: Optional[Path] = None):
        self._model = None
        self._model_path = model_path if model_path is not None else _resolve_model_path()
        self._model_version = os.environ.get("MODEL_VERSION", "default").strip() or "default"
        self._feature_mismatch = False
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
                self._warmup_model()
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

    def _warmup_model(self) -> None:
        """Run a dummy prediction to initialize BLAS/OpenMP thread pools.

        Without this, the FIRST real query triggers thread pool initialization
        for all 16 CPU cores, which can take 30-60 seconds on first call.
        Warming up at startup moves this cost to startup time, not inference time.
        """
        if self._model is None:
            return
        import time
        try:
            # Resolve the model's expected feature dimension robustly.
            # Try in order:
            #   1. pipeline.n_features_in_                  (sklearn Pipeline)
            #   2. first step (e.g. StandardScaler).n_features_in_
            #   3. final estimator's first sub-estimator    (VotingClassifier path)
            #   4. fallback to len(extract_chunk_features(empty)) → current schema
            n_features = None
            try:
                n_features = int(self._model.n_features_in_)
            except AttributeError:
                pass
            if n_features is None:
                try:
                    first_step = next(iter(self._model.named_steps.values()))
                    n_features = int(first_step.n_features_in_)
                except (AttributeError, StopIteration):
                    pass
            if n_features is None:
                try:
                    clf = self._model.named_steps.get("clf")
                    inner = clf.estimators_[0] if hasattr(clf, "estimators_") else clf
                    n_features = int(inner.n_features_in_)
                except (AttributeError, IndexError):
                    pass
            if n_features is None:
                # Last resort: derive from current feature extractor (always
                # consistent with the running features.py).
                from poker44.miner_model.features import extract_chunk_features
                n_features = int(extract_chunk_features([]).shape[0])

            dummy = np.zeros((4, n_features), dtype=np.float32)
            t0 = time.monotonic()
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _limit_threads()
                self._model.predict_proba(dummy)
            elapsed = time.monotonic() - t0
            bt.logging.info(
                f"[BotDetector] Model warmed up in {elapsed:.2f}s "
                f"(n_features={n_features})"
            )
        except Exception as exc:
            bt.logging.warning(f"[BotDetector] Warm-up failed (non-fatal): {exc}")

    def is_model_loaded(self) -> bool:
        return self._model is not None and not self._feature_mismatch

    @property
    def model_label(self) -> str:
        """Short label for logs: 'ML(v3_gb_mixed)' or 'heuristic'."""
        if self._model is not None and not self._feature_mismatch:
            return f"ML({self._model_version})"
        if self._feature_mismatch:
            return f"heuristic(mismatch:{self._model_version})"
        return "heuristic"

    def score_chunk(self, chunk: List[Dict[str, Any]]) -> float:
        """Return bot-risk score in [0, 1]. Validator rounds at 0.5 for binary metrics."""
        if not chunk:
            return 0.5
        if self._model is not None and not self._feature_mismatch:
            return self._score_with_model(chunk)
        return self._score_heuristic(chunk)

    def score_chunks_batch(self, chunks: List[List[Dict[str, Any]]]) -> List[float]:
        """Score all chunks in one batched predict_proba call.

        Much faster than calling score_chunk() in a loop because the model
        processes a (N, 100) matrix in a single pass instead of N separate
        (1, 100) calls through the CalibratedClassifierCV wrapper.
        Limits BLAS/OpenMP threads to 1 to avoid thread-pool overhead on
        high-core-count CPUs that makes small-matrix inference take 60+ seconds.
        """
        if not chunks:
            return []

        if self._model is None or self._feature_mismatch:
            return [self._score_heuristic(chunk) for chunk in chunks]

        try:
            import time
            import warnings
            is_seq = bool(getattr(self._model, "_is_sequence_model", False))
            t_feat = time.monotonic()
            if is_seq:
                # Sequence model consumes raw chunks directly; feature extraction
                # happens inside the wrapper (per-hand matrix + padding).
                model_input: Any = chunks
            else:
                model_input = np.stack([
                    extract_chunk_features(chunk) for chunk in chunks
                ])
            t_pred = time.monotonic()
            with warnings.catch_warnings(), _limit_threads():
                warnings.simplefilter("ignore")
                probs = self._model.predict_proba(model_input)[:, 1]
            t_done = time.monotonic()
            bt.logging.debug(
                f"[BotDetector] feat={t_pred-t_feat:.3f}s "
                f"predict={t_done-t_pred:.3f}s "
                f"n={len(chunks)} seq={is_seq}"
            )
            return [float(np.clip(p - _SCORE_BIAS, 0.0, 1.0)) for p in probs]
        except ValueError as exc:
            if not self._feature_mismatch:
                bt.logging.warning(
                    f"[BotDetector] Feature dimension mismatch in batch call "
                    f"(version={self._model_version!r}): {exc}. "
                    f"Falling back to heuristic for all chunks."
                )
                self._feature_mismatch = True
            return [self._score_heuristic(chunk) for chunk in chunks]
        except Exception as exc:
            bt.logging.error(f"[BotDetector] Batch scoring error: {exc}")
            return [self._score_heuristic(chunk) for chunk in chunks]

    def predict_chunk(self, chunk: List[Dict[str, Any]]) -> bool:
        """Binary prediction (used for synapse.predictions — note: validator ignores this)."""
        return self.score_chunk(chunk) >= 0.5

    # ------------------------------------------------------------------
    # ML path
    # ------------------------------------------------------------------

    def _score_with_model(self, chunk: List[Dict[str, Any]]) -> float:
        is_seq = bool(getattr(self._model, "_is_sequence_model", False))
        if is_seq:
            model_input: Any = [chunk]
        else:
            model_input = extract_chunk_features(chunk).reshape(1, -1)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prob = float(self._model.predict_proba(model_input)[0, 1])
        except ValueError as exc:
            # Feature dimension mismatch: old model trained on different feature count.
            # Fall back to heuristic and warn once.
            if not self._feature_mismatch:
                bt.logging.warning(
                    f"[BotDetector] Feature dimension mismatch for "
                    f"version={self._model_version!r}: {exc}. "
                    f"Falling back to heuristic. Retrain with: "
                    f"python -m poker44.miner_model.train --version v3_gb_mixed"
                )
                self._feature_mismatch = True
            return self._score_heuristic(chunk)
        except AttributeError:
            prob = float(self._model.predict(model_input)[0])

        # Apply conservative bias: push borderline scores below the validator's
        # 0.5 rounding threshold to reduce false positives on human chunks.
        return float(np.clip(prob - _SCORE_BIAS, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Heuristic fallback (used before training or on dimension mismatch)
    #
    # Key insight: bots are consistent → low within-chunk variance of depth.
    # Humans are diverse → high within-chunk variance of depth.
    # ------------------------------------------------------------------

    def _score_heuristic(self, chunk: List[Dict[str, Any]]) -> float:
        from collections import Counter

        depths: List[float] = []
        postflop_fracs: List[float] = []
        bet_cvs: List[float] = []

        for hand in chunk:
            actions = hand.get("actions") or []

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

            amounts = [
                float(a.get("normalized_amount_bb") or 0.0)
                for a in actions
                if float(a.get("normalized_amount_bb") or 0.0) > 0.0
            ]
            if len(amounts) > 1:
                m = float(np.mean(amounts))
                s = float(np.std(amounts))
                bet_cvs.append(min(s / max(m, 1e-6), 5.0))

        n = len(depths)
        if n == 0:
            return 0.5

        mean_depth    = float(np.mean(depths))
        std_depth     = float(np.std(depths))
        mean_postflop = float(np.mean(postflop_fracs))
        std_postflop  = float(np.std(postflop_fracs))

        # Low bet_cv → more mechanical sizing → more bot-like
        mean_bet_cv = float(np.mean(bet_cvs)) if bet_cvs else 1.0
        bet_cv_signal = max(0.0, 1.0 - mean_bet_cv * 0.4)

        depth_signal         = mean_depth
        consistency_signal   = max(0.0, 1.0 - std_depth * 4.0)
        postflop_consistency = max(0.0, 1.0 - std_postflop * 3.0)

        score = (
            0.25 * depth_signal
            + 0.30 * consistency_signal
            + 0.20 * postflop_consistency
            + 0.25 * bet_cv_signal
        )
        # Apply same bias as ML path for consistency
        return float(np.clip(score - _SCORE_BIAS, 0.0, 1.0))
