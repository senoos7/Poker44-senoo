"""
Training script for the Poker44 bot-detection model.

Supports a versioned model history so you can A/B test improvements
without overwriting previous models. Each version is saved to:

    poker44/miner_model/models/<version>/model.pkl
    poker44/miner_model/models/<version>/metadata.json

The default model.pkl (used when MODEL_VERSION is unset) is updated
only when you explicitly pass --update-default.

──────────────────────────────────────────────────────────────────────
Version roadmap
──────────────────────────────────────────────────────────────────────
  v1_rf_synthetic  — baseline (archived; trained on 60 features)
    data: custom synthetic bots, no anti-shortcut filter
    model: RandomForest + CalibratedCV

  v2_rf_mixed      — Phase 1: fix training data source (archived; trained on 60 features)
    data: build_mixed_labeled_chunks() (same as validator, anti-shortcut filtered)
    model: RandomForest + CalibratedCV (same architecture, only data changes)

  v3_gb_mixed      — Phase 2: better model + richer features
    data: build_mixed_labeled_chunks()
    model: HistGradientBoostingClassifier + 76-feature extraction
    why: HGBM handles feature interactions natively, reduces variance,
         produces better-calibrated probabilities, and is more stable across
         diverse validator batches than RF. Addresses score instability.

  v5_hgbm_enhanced — Phase 3: enhanced features + HistGBM
    data: build_mixed_labeled_chunks() (anti-shortcut filtered)
    model: HistGradientBoostingClassifier + 100-feature extraction (25 per-hand)
    new features vs v3/v4:
      - unique_bet_ratio, preflop_raise_frac, max_amount_bb, blind_frac,
        call_raise_ratio, n_actions_norm
    why: v1 uses real hands from live benchmark tables. Trained on synthetic
         data, so CV=1.0 is meaningless — high sim-to-real gap expected.

  v6_benchmark — Phase 4: real evaluation data
    data: public benchmark API (https://api.poker44.net/api/v1/benchmark)
          real labeled chunks from live v1 evaluation, already in miner-visible
          format. No synthetic generation needed.
    model: HistGradientBoostingClassifier + CalibratedCV(isotonic) + 100 features
    why: eliminates the sim-to-real gap entirely.
    issue: isotonic calibration collapses probabilities to near-binary (0.94 or 0.0)
           which limits within-class ranking and caps composite score ~0.51.
    download first: python scripts/download_benchmark.py

  v7_sigmoid_calib — Phase 5: sigmoid calibration for graded scores (RECOMMENDED)
    data: same benchmark API data as v6
    model: HistGradientBoostingClassifier + CalibratedCV(sigmoid) + 100 features
    key change: sigmoid (Platt scaling) replaces isotonic → smooth probability
                curve → scores like 0.7, 0.8, 0.9 instead of always 0.94/0.0
    why: graded scores allow within-class RANKING which drives AUROC/AP improvement.
         Higher composite score → top-3 contention → actual on-chain incentive.
    download first: python scripts/download_benchmark.py

──────────────────────────────────────────────────────────────────────
Usage
──────────────────────────────────────────────────────────────────────
  # Phase 5 — sigmoid calibrated model (RECOMMENDED for new miners)
  python scripts/download_benchmark.py           # refresh benchmark data
  python -m poker44.miner_model.train --version v7_sigmoid_calib --data-source benchmark

  # Phase 4 — isotonic (current default, kept for comparison)
  python -m poker44.miner_model.train --version v6_benchmark --data-source benchmark

  # After evaluating, promote a version to the default slot
  python -m poker44.miner_model.train --version v7_sigmoid_calib --update-default
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker44.miner_model.features import extract_chunk_features, CHUNK_FEATURE_NAMES
from poker44.miner_model.sanitize import sanitize_hand

# ------------------------------------------------------------------
# Paths & defaults
# ------------------------------------------------------------------
_MODEL_DIR        = Path(__file__).parent
_MODELS_ROOT      = _MODEL_DIR / "models"
_DEFAULT_CORPUS   = REPO_ROOT / "hands_generator" / "human_hands" / "poker_hands_combined.json.gz"
_DEFAULT_OUTPUT   = _MODEL_DIR / "model.pkl"   # legacy default slot
_DEFAULT_N_CHUNKS = 4000    # total labeled chunks (half bot, half human); 4000 for v3 stability
_FAST_N_CHUNKS    = 800
_CHUNK_SIZE_MIN   = 60
_CHUNK_SIZE_MAX   = 120


# ------------------------------------------------------------------
# Extended bot profiles (used only for synthetic data source)
# ------------------------------------------------------------------

def _extended_bot_profiles():
    from hands_generator.bot_hands.generate_poker_data import BotProfile
    return [
        BotProfile(name="balanced",         tightness=0.54, aggression=0.66, bluff_freq=0.05),
        BotProfile(name="tight_aggressive", tightness=0.62, aggression=0.80, bluff_freq=0.04),
        BotProfile(name="loose_aggressive", tightness=0.44, aggression=0.78, bluff_freq=0.08),
        BotProfile(name="tight_passive",    tightness=0.60, aggression=0.50, bluff_freq=0.02),
        BotProfile(name="loose_passive",    tightness=0.46, aggression=0.46, bluff_freq=0.05),
        BotProfile(name="nit",              tightness=0.82, aggression=0.30, bluff_freq=0.01),
        BotProfile(name="maniac",           tightness=0.28, aggression=0.92, bluff_freq=0.15),
        BotProfile(name="calling_station",  tightness=0.38, aggression=0.18, bluff_freq=0.03),
        BotProfile(name="nit_aggressive",   tightness=0.80, aggression=0.88, bluff_freq=0.03),
        BotProfile(name="fish",             tightness=0.32, aggression=0.55, bluff_freq=0.12),
        BotProfile(name="semi_gto",         tightness=0.50, aggression=0.72, bluff_freq=0.07),
        BotProfile(name="reg_tight",        tightness=0.68, aggression=0.74, bluff_freq=0.05),
        BotProfile(name="reg_loose",        tightness=0.48, aggression=0.70, bluff_freq=0.06),
        BotProfile(name="ultra_tight",      tightness=0.88, aggression=0.60, bluff_freq=0.02),
        BotProfile(name="hyper_aggressive", tightness=0.36, aggression=0.95, bluff_freq=0.18),
    ]


# ------------------------------------------------------------------
# Data source 1: synthetic (original approach)
# ------------------------------------------------------------------

def _load_human_corpus(path: Path) -> List[Dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        hands = json.load(f)
    print(f"  Loaded {len(hands):,} human hands from {path.name}")
    return hands


def _generate_bot_chunks_synthetic(
    n_chunks: int,
    rng: random.Random,
) -> List[List[Dict[str, Any]]]:
    from hands_generator.data_generator import generate_bot_chunk
    profiles = _extended_bot_profiles()
    chunks = []
    for i in range(n_chunks):
        size = rng.randint(_CHUNK_SIZE_MIN, _CHUNK_SIZE_MAX)
        chunk = generate_bot_chunk(
            size=size,
            profiles=profiles,
            seed=rng.randint(0, 10**9),
        )
        chunks.append(chunk)
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{n_chunks} bot chunks...")
    return chunks


def _sample_human_chunks(
    human_hands: List[Dict[str, Any]],
    n_chunks: int,
    rng: random.Random,
) -> List[List[Dict[str, Any]]]:
    chunks = []
    for _ in range(n_chunks):
        size = rng.randint(_CHUNK_SIZE_MIN, _CHUNK_SIZE_MAX)
        size = min(size, len(human_hands))
        chunks.append(rng.sample(human_hands, size))
    return chunks


def _generate_synthetic_data(
    n_per_class: int,
    human_corpus: Path,
    rng: random.Random,
) -> Tuple[List[List[Dict]], List[List[Dict]]]:
    """Generate raw bot + human chunks using our custom synthetic approach."""
    print(f"\n[A] Loading human corpus...")
    human_hands = _load_human_corpus(human_corpus)

    print(f"\n[B] Generating {n_per_class} synthetic bot chunks...")
    bot_chunks_raw = _generate_bot_chunks_synthetic(n_per_class, rng)

    print(f"\n[C] Sampling {n_per_class} human chunks...")
    human_chunks_raw = _sample_human_chunks(human_hands, n_per_class, rng)

    return bot_chunks_raw, human_chunks_raw


# ------------------------------------------------------------------
# Data source 2: mixed (Phase 1 upgrade — uses validator's own generator)
# ------------------------------------------------------------------

def _generate_mixed_data(
    n_per_class: int,
    human_corpus: Path,
    rng: random.Random,
    fast: bool = False,
) -> Tuple[List[List[Dict]], List[List[Dict]]]:
    """
    Generate training data using build_mixed_labeled_chunks() — the same
    function the validator uses. Key benefit: the bot chunks selected here
    have already passed the anti-shortcut filter (no single feature achieves
    >70% accuracy), so the model learns to detect *hard* bots, not easy ones.

    Strategy:
      - Call build_mixed_labeled_chunks() in batches of chunk_count=80 chunks.
      - Accumulate bot and human chunks separately until we have n_per_class each.
      - Use different window_ids to get diverse data across batches.
    """
    from hands_generator.mixed_dataset_provider import (
        build_mixed_labeled_chunks,
        MixedDatasetConfig,
    )

    # Fewer rounds / candidates for training speed (validator uses 4 rounds, 8 candidates)
    # We use 2 rounds, 6 candidates — still good quality but 2-3x faster
    bot_generation_rounds = 1 if fast else 2
    bot_candidate_attempts = 4 if fast else 6

    # ~80 chunks per call (40 bot + 40 human), need ceil(n_per_class/40) calls
    chunks_per_call = 80
    calls_needed = max(1, math.ceil(n_per_class / (chunks_per_call // 2)))

    print(f"\n[A] Generating {n_per_class} anti-shortcut-filtered bot chunks")
    print(f"    + {n_per_class} human chunks via build_mixed_labeled_chunks()")
    print(f"    Batches needed: ~{calls_needed}  (rounds/batch={bot_generation_rounds}, "
          f"candidates/chunk={bot_candidate_attempts})")
    print(f"    Estimated time: {calls_needed * 2}-{calls_needed * 5} minutes")

    bot_chunks_raw: List[List[Dict]] = []
    human_chunks_raw: List[List[Dict]] = []

    for call_idx in range(calls_needed * 2):  # extra headroom
        if len(bot_chunks_raw) >= n_per_class and len(human_chunks_raw) >= n_per_class:
            break

        cfg = MixedDatasetConfig(
            human_json_path=human_corpus,
            chunk_count=chunks_per_call,
            min_hands_per_chunk=_CHUNK_SIZE_MIN,
            max_hands_per_chunk=_CHUNK_SIZE_MAX,
            human_ratio=0.5,
            seed=rng.randint(0, 10**6),
            bot_candidate_attempts_per_chunk=bot_candidate_attempts,
            max_bot_generation_rounds=bot_generation_rounds,
            max_shortcut_rule_accuracy=0.72,  # slightly looser than validator's 0.70
        )
        try:
            labeled_chunks, _, stats = build_mixed_labeled_chunks(cfg, window_id=call_idx)
        except Exception as exc:
            print(f"  Warning: batch {call_idx + 1} failed: {exc}. Skipping.")
            continue

        for chunk in labeled_chunks:
            hands = chunk.get("hands", [])
            if not hands:
                continue
            if chunk.get("is_bot"):
                if len(bot_chunks_raw) < n_per_class:
                    bot_chunks_raw.append(hands)
            else:
                if len(human_chunks_raw) < n_per_class:
                    human_chunks_raw.append(hands)

        bot_have   = min(len(bot_chunks_raw),   n_per_class)
        human_have = min(len(human_chunks_raw), n_per_class)
        print(f"  Batch {call_idx + 1}: bot={bot_have}/{n_per_class}  "
              f"human={human_have}/{n_per_class}  "
              f"[shortcut_acc={stats.get('shortcut_rule_accuracy', '?'):.3f}]")

    bot_chunks_raw   = bot_chunks_raw[:n_per_class]
    human_chunks_raw = human_chunks_raw[:n_per_class]

    if len(bot_chunks_raw) < n_per_class or len(human_chunks_raw) < n_per_class:
        print(f"\n  WARNING: only collected {len(bot_chunks_raw)} bot / "
              f"{len(human_chunks_raw)} human chunks (target={n_per_class}).")
        print(f"  Proceeding with available data.")

    return bot_chunks_raw, human_chunks_raw


# ------------------------------------------------------------------
# Sanitization + feature matrix
# ------------------------------------------------------------------

def _apply_sanitization(chunks: List[List[Dict]]) -> List[List[Dict]]:
    return [[sanitize_hand(h) for h in chunk] for chunk in chunks]


_DEFAULT_BENCHMARK_PATH = Path(__file__).resolve().parents[2] / "data" / "benchmark.json.gz"


def _load_benchmark_data(
    benchmark_path: Optional[Path] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[List[Dict]], List[List[Dict]]]:
    """Load real labeled chunks from the downloaded benchmark file.

    Returns (bot_chunks, human_chunks). Benchmark chunks are already in
    miner-visible format (no further sanitization required).
    """
    path = Path(benchmark_path or _DEFAULT_BENCHMARK_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {path}\n"
            f"Download it first:\n"
            f"  python scripts/download_benchmark.py --out {path}"
        )

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        data = json.load(f)

    chunks_raw = data["chunks"]
    labels = data["groundTruth"]

    if len(chunks_raw) != len(labels):
        raise ValueError(f"chunks/labels length mismatch: {len(chunks_raw)} vs {len(labels)}")

    bot_chunks   = [c for c, lbl in zip(chunks_raw, labels) if lbl == 1]
    human_chunks = [c for c, lbl in zip(chunks_raw, labels) if lbl == 0]

    meta = data.get("meta", {})
    print(f"  Benchmark: {len(bot_chunks)} bot + {len(human_chunks)} human chunks "
          f"| {meta.get('total_hands', '?'):,} hands "
          f"| dates={meta.get('dates', '?')}")

    if rng is not None:
        rng.shuffle(bot_chunks)
        rng.shuffle(human_chunks)

    return bot_chunks, human_chunks


def build_training_matrix(
    bot_chunks: List[List[Dict]],
    human_chunks: List[List[Dict]],
) -> Tuple[np.ndarray, np.ndarray]:
    all_chunks = bot_chunks + human_chunks
    labels = np.array([1] * len(bot_chunks) + [0] * len(human_chunks), dtype=int)
    X = np.vstack([extract_chunk_features(c) for c in all_chunks])
    idx = np.random.default_rng(42).permutation(len(labels))
    return X[idx], labels[idx]


# ------------------------------------------------------------------
# Model training
# ------------------------------------------------------------------

def train_model(X: np.ndarray, y: np.ndarray, version: str = "") -> Any:
    """Dispatch to the appropriate model architecture based on version name."""
    if version.startswith("v9") or "hero" in version:
        return _train_model_v9_hero(X, y)
    if (version.startswith("v8") or "structured" in version or "ensemble" in version):
        return _train_model_v8_structured(X, y)
    if version.startswith("v7") or "sigmoid" in version or "direct" in version:
        return _train_model_v7_sigmoid(X, y)
    if _is_hgbm_version(version):
        return _train_model_v3_hgbm(X, y)
    return _train_model_v2_rf(X, y)


def _train_model_v2_rf(X: np.ndarray, y: np.ndarray):
    """
    RandomForest + CalibratedClassifierCV pipeline (v1/v2 architecture).
    Kept for backward compatibility and A/B comparison.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    n_samples, n_features = X.shape
    print(f"\n  [RF] Training on {n_samples} chunks × {n_features} features")
    print(f"  Class balance: {y.sum()} bot / {(1-y).sum()} human")

    base = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(base, cv=3, method="isotonic")),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\n  Running 5-fold cross-validation (average_precision)...")
    cv_ap  = cross_val_score(model, X, y, cv=cv, scoring="average_precision", n_jobs=1)
    cv_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy",           n_jobs=1)
    print(f"  CV AP:     {cv_ap.mean():.4f} ± {cv_ap.std():.4f}  (target >0.80)")
    print(f"  CV Acc:    {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

    print("\n  Fitting final model on full dataset...")
    model.fit(X, y)

    # Feature importance from the first calibrated estimator
    rf = model.named_steps["clf"].calibrated_classifiers_[0].estimator
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    print("\n  Top 10 features by importance:")
    top_features = []
    for rank, idx in enumerate(top_idx, 1):
        name = CHUNK_FEATURE_NAMES[idx] if idx < len(CHUNK_FEATURE_NAMES) else f"feat_{idx}"
        print(f"    {rank:>2}. {name:<35} {importances[idx]:.4f}")
        top_features.append({"rank": rank, "name": name, "importance": round(float(importances[idx]), 4)})

    # Set n_jobs=1 before pickling to avoid joblib conflicts in async miner
    clf = model.named_steps["clf"]
    for cal_clf in clf.calibrated_classifiers_:
        if hasattr(cal_clf, "estimator"):
            cal_clf.estimator.n_jobs = 1
        if hasattr(cal_clf, "base_estimator"):
            cal_clf.base_estimator.n_jobs = 1
    print("\n  Set n_jobs=1 on inner RandomForests (safe for async inference).")

    return model, cv_ap, cv_acc, top_features


def _train_model_v3_hgbm(X: np.ndarray, y: np.ndarray):
    """
    HistGradientBoostingClassifier + CalibratedClassifierCV pipeline (v3 architecture).

    Why HGBM over RandomForest:
    - Handles feature interactions natively via gradient boosting trees
    - Lower variance across diverse validator batches (key for stability)
    - Better probability calibration for borderline chunks near 0.5
    - Native regularization (l2, min_samples_leaf) avoids overfitting to training bots
    - Faster training at larger data sizes

    The combination of HGBM + isotonic calibration produces scores that are
    more confidently separated from the 0.5 decision boundary, reducing FPR spikes.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    n_samples, n_features = X.shape
    print(f"\n  [HGBM v3] Training on {n_samples} chunks × {n_features} features")
    print(f"  Class balance: {y.sum()} bot / {(1-y).sum()} human")

    # HGBM is scale-invariant (tree-based), but StandardScaler doesn't hurt
    # and keeps the pipeline contract consistent with v1/v2 for scoring.
    base = HistGradientBoostingClassifier(
        max_iter=200,           # 200 vs 400: ~2x faster inference, still strong accuracy
        max_depth=5,            # shallower trees → faster prediction per sample
        min_samples_leaf=16,    # more regularization for generalization
        learning_rate=0.06,     # slightly faster convergence to compensate fewer iters
        l2_regularization=0.3,
        max_bins=63,            # 63 vs 127: ~1.5x faster per-sample prediction
        class_weight="balanced",
        random_state=42,
        early_stopping=False,
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(base, cv=3, method="isotonic")),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\n  Running 5-fold cross-validation (average_precision)...")
    cv_ap  = cross_val_score(model, X, y, cv=cv, scoring="average_precision", n_jobs=1)
    cv_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy",           n_jobs=1)
    print(f"  CV AP:     {cv_ap.mean():.4f} ± {cv_ap.std():.4f}  (target >0.85)")
    print(f"  CV Acc:    {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

    # Calibration quality: report mean predicted probability for each class
    # (well-calibrated: bot scores high, human scores low, few near 0.5)
    from sklearn.model_selection import cross_val_predict
    cv_probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    bot_scores   = cv_probs[y == 1]
    human_scores = cv_probs[y == 0]
    print(f"\n  Calibration check (lower ambiguity = better FPR stability):")
    print(f"    Bot chunks:   mean={bot_scores.mean():.3f}  std={bot_scores.std():.3f}"
          f"  pct_above_0.5={100*np.mean(bot_scores > 0.5):.1f}%")
    print(f"    Human chunks: mean={human_scores.mean():.3f}  std={human_scores.std():.3f}"
          f"  pct_below_0.5={100*np.mean(human_scores < 0.5):.1f}%")
    ambiguous = np.mean((cv_probs > 0.44) & (cv_probs < 0.56))
    print(f"    Ambiguous (0.44–0.56): {100*ambiguous:.1f}%  (target <10%)")

    print("\n  Fitting final model on full dataset...")
    model.fit(X, y)

    # Feature importance from HGBM via permutation importance (sklearn ≥1.2 has
    # feature_importances_ natively; older versions require permutation approach).
    top_features = []
    try:
        hgbm = model.named_steps["clf"].calibrated_classifiers_[0].estimator
        if hasattr(hgbm, "feature_importances_"):
            importances = hgbm.feature_importances_
        else:
            # Fallback: use a small held-out sample for permutation importance
            from sklearn.inspection import permutation_importance as _perm_imp
            X_sample = X[:500]
            y_sample = y[:500]
            _pi = _perm_imp(model, X_sample, y_sample, n_repeats=5,
                            scoring="average_precision", random_state=42, n_jobs=1)
            importances = _pi.importances_mean
        top_idx = np.argsort(importances)[::-1][:10]
        print("\n  Top 10 features by importance:")
        for rank, idx in enumerate(top_idx, 1):
            name = CHUNK_FEATURE_NAMES[idx] if idx < len(CHUNK_FEATURE_NAMES) else f"feat_{idx}"
            print(f"    {rank:>2}. {name:<35} {importances[idx]:.4f}")
            top_features.append({"rank": rank, "name": name, "importance": round(float(importances[idx]), 4)})
    except Exception as exc:
        print(f"  (Feature importance skipped: {exc})")

    return model, cv_ap, cv_acc, top_features


def _train_model_v7_sigmoid(X: np.ndarray, y: np.ndarray):
    """
    Shallow HistGBM WITHOUT calibration wrapper (v7 architecture).

    Root cause of binary output in v6:
    - Both isotonic AND sigmoid calibration collapse to near-binary when the
      underlying model perfectly separates training data (CV acc=1.0).
    - When raw logit = ±10, sigmoid(10) = 0.99995 → still binary.

    Fix: use raw HistGBM predict_proba WITHOUT CalibratedClassifierCV.
    HistGBM leaf proportions are naturally more graded when trees are SHALLOW:
    - max_depth=2: each leaf covers a wide region → many training samples per leaf
      → leaf probability = fraction of bots in that leaf = spread over (0, 1)
    - max_iter=50: intentionally underfit so uncertain cases stay in [0.3, 0.7]
    - min_samples_leaf=80: requires 80 samples per leaf → further forces grading

    Trade-off: slightly lower accuracy on obvious cases, but MUCH better AUROC/AP
    on borderline chunks because the validator can rank partial confidence correctly.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    n_samples, n_features = X.shape
    print(f"\n  [HGBM v7 uncalibrated-shallow] Training on {n_samples} chunks × {n_features} features")
    print(f"  Class balance: {y.sum()} bot / {(1-y).sum()} human")

    # Intentionally shallow / underfit → graded probabilities on live data
    base = HistGradientBoostingClassifier(
        max_iter=80,
        max_depth=2,
        min_samples_leaf=80,
        learning_rate=0.15,
        l2_regularization=1.0,
        max_bins=31,
        class_weight="balanced",
        random_state=42,
        early_stopping=False,
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", base),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\n  Running 5-fold cross-validation (average_precision)...")
    cv_ap  = cross_val_score(model, X, y, cv=cv, scoring="average_precision", n_jobs=1)
    cv_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy",           n_jobs=1)
    print(f"  CV AP:     {cv_ap.mean():.4f} ± {cv_ap.std():.4f}")
    print(f"  CV Acc:    {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

    from sklearn.model_selection import cross_val_predict
    cv_probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    bot_scores   = cv_probs[y == 1]
    human_scores = cv_probs[y == 0]
    print(f"\n  Score distribution (want spread, not just 0/1):")
    print(f"    Bot chunks:   mean={bot_scores.mean():.3f}  std={bot_scores.std():.3f}"
          f"  pct_above_0.5={100*np.mean(bot_scores > 0.5):.1f}%")
    print(f"    Human chunks: mean={human_scores.mean():.3f}  std={human_scores.std():.3f}"
          f"  pct_below_0.5={100*np.mean(human_scores < 0.5):.1f}%")
    ambiguous = np.mean((cv_probs > 0.3) & (cv_probs < 0.7))
    print(f"    In 0.3–0.7 range: {100*ambiguous:.1f}%  (shows graded spread)")

    print("\n  Fitting final model on full dataset...")
    model.fit(X, y)

    top_features = []
    try:
        hgbm = model.named_steps["clf"]
        if hasattr(hgbm, "feature_importances_"):
            importances = hgbm.feature_importances_
            top_idx = np.argsort(importances)[::-1][:10]
            print("\n  Top 10 features by importance:")
            for rank, idx in enumerate(top_idx, 1):
                name = CHUNK_FEATURE_NAMES[idx] if idx < len(CHUNK_FEATURE_NAMES) else f"feat_{idx}"
                print(f"    {rank:>2}. {name:<35} {importances[idx]:.4f}")
                top_features.append({"rank": rank, "name": name, "importance": round(float(importances[idx]), 4)})
    except Exception as exc:
        print(f"  (Feature importance skipped: {exc})")

    return model, cv_ap, cv_acc, top_features


def _train_model_v8_structured(X: np.ndarray, y: np.ndarray):
    """
    Soft-voting ensemble: HistGBM + ExtraTrees + LogisticRegression.

    Why ensemble (vs a single model):
    - Top-ranked competitor miners use 'structured-action-chunk-detector' and
      'extra-trees' style models. ExtraTrees randomizes splits, which gives
      better generalization on distribution-shifted live data.
    - LogisticRegression on standardized features acts as a smooth, calibrated
      tie-breaker, mapping each tree's hard split into a graded probability —
      this is the key to escaping the bimodal 0.94 / 0.0 trap.
    - Averaging three models with different inductive biases reduces variance
      and produces well-spread probabilities (target: 0.2–0.8 for ambiguous
      chunks, instead of binary).

    Designed to consume the v8 feature set (34 per-hand × 4 = 136 features)
    which now includes structural / sequence signals (bigram diversity,
    actor concentration, pot-relative sizing, repeat amounts).
    """
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        ExtraTreesClassifier,
        VotingClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    n_samples, n_features = X.shape
    print(f"\n  [v8 structured ensemble] Training on {n_samples} chunks × {n_features} features")
    print(f"  Class balance: {y.sum()} bot / {(1-y).sum()} human")

    hgbm = HistGradientBoostingClassifier(
        max_iter=160,
        max_depth=4,
        min_samples_leaf=40,
        learning_rate=0.07,
        l2_regularization=0.5,
        max_bins=63,
        class_weight="balanced",
        random_state=42,
        early_stopping=False,
    )
    et = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=14,
        min_samples_leaf=4,
        max_features="sqrt",
        bootstrap=False,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    lr = LogisticRegression(
        C=0.5,
        max_iter=500,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )

    voter = VotingClassifier(
        estimators=[("hgbm", hgbm), ("et", et), ("lr", lr)],
        voting="soft",
        weights=[2.0, 2.0, 1.0],
        n_jobs=1,
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", voter),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\n  Running 5-fold cross-validation (average_precision)...")
    cv_ap  = cross_val_score(model, X, y, cv=cv, scoring="average_precision", n_jobs=1)
    cv_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy",           n_jobs=1)
    print(f"  CV AP:     {cv_ap.mean():.4f} ± {cv_ap.std():.4f}")
    print(f"  CV Acc:    {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

    from sklearn.model_selection import cross_val_predict
    cv_probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    bot_scores   = cv_probs[y == 1]
    human_scores = cv_probs[y == 0]
    print(f"\n  Score spread (want graded, not binary):")
    print(f"    Bot chunks:   mean={bot_scores.mean():.3f}  std={bot_scores.std():.3f}"
          f"  pct_above_0.5={100*np.mean(bot_scores > 0.5):.1f}%")
    print(f"    Human chunks: mean={human_scores.mean():.3f}  std={human_scores.std():.3f}"
          f"  pct_below_0.5={100*np.mean(human_scores < 0.5):.1f}%")
    in_band = np.mean((cv_probs > 0.2) & (cv_probs < 0.8))
    print(f"    In 0.2–0.8 range: {100*in_band:.1f}%  (graded chunks → better AUROC)")

    print("\n  Fitting final model on full dataset...")
    model.fit(X, y)

    # Feature importance from ExtraTrees (most interpretable of the three)
    top_features = []
    try:
        et_fitted = model.named_steps["clf"].named_estimators_["et"]
        if hasattr(et_fitted, "feature_importances_"):
            importances = et_fitted.feature_importances_
            top_idx = np.argsort(importances)[::-1][:12]
            print("\n  Top 12 features by ExtraTrees importance:")
            for rank, idx in enumerate(top_idx, 1):
                name = CHUNK_FEATURE_NAMES[idx] if idx < len(CHUNK_FEATURE_NAMES) else f"feat_{idx}"
                print(f"    {rank:>2}. {name:<35} {importances[idx]:.4f}")
                top_features.append(
                    {"rank": rank, "name": name, "importance": round(float(importances[idx]), 4)}
                )
    except Exception as exc:
        print(f"  (Feature importance skipped: {exc})")

    return model, cv_ap, cv_acc, top_features


# ------------------------------------------------------------------
# Metadata save/load
# ------------------------------------------------------------------

def _save_metadata(
    path: Path,
    version: str,
    data_source: str,
    n_per_class: int,
    cv_ap: np.ndarray,
    cv_acc: np.ndarray,
    top_features: list,
    training_seconds: float,
    n_features: int,
    n_bot_chunks: Optional[int] = None,
    n_human_chunks: Optional[int] = None,
) -> None:
    _n_bot   = n_bot_chunks   if n_bot_chunks   is not None else n_per_class
    _n_human = n_human_chunks if n_human_chunks is not None else n_per_class
    meta = {
        "version": version,
        "trained_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_source": data_source,
        "n_chunks": _n_bot + _n_human,
        "n_bot_chunks": _n_bot,
        "n_human_chunks": _n_human,
        "chunk_size_range": [_CHUNK_SIZE_MIN, _CHUNK_SIZE_MAX],
        "model_type": (
            "Pipeline(StandardScaler + CalibratedClassifierCV(HistGradientBoosting(depth=4, hero-aware), sigmoid, cv=3))"
            if (version.startswith("v9") or "hero" in version) else
            "Pipeline(StandardScaler + Voting(HistGBM + ExtraTrees + LogReg))"
            if (version.startswith("v8") or "structured" in version or "ensemble" in version) else
            "Pipeline(StandardScaler + HistGradientBoosting(depth=2, uncalibrated))"
            if (version.startswith("v7") or "sigmoid" in version or "direct" in version) else
            "Pipeline(StandardScaler + CalibratedClassifierCV(HistGradientBoosting, isotonic, cv=3))"
            if _is_hgbm_version(version) else
            "Pipeline(StandardScaler + CalibratedClassifierCV(RandomForest, isotonic, cv=3))"
        ),
        "n_features": n_features,
        "cv_ap_mean": round(float(cv_ap.mean()), 4),
        "cv_ap_std":  round(float(cv_ap.std()),  4),
        "cv_acc_mean": round(float(cv_acc.mean()), 4),
        "cv_acc_std":  round(float(cv_acc.std()),  4),
        "training_time_seconds": round(training_seconds, 1),
        "top_features": top_features,
        "description": {
            "synthetic": (
                "Custom synthetic bots (15 profiles). Training data NOT filtered by "
                "anti-shortcut mechanism. Model picks up simple patterns that the "
                "validator actively removes."
            ),
            "mixed": (
                "Uses build_mixed_labeled_chunks() — the same generator the validator "
                "uses. Bot chunks passed the anti-shortcut filter (no single feature "
                "achieves >70% accuracy). Model must learn subtle, high-order signals."
            ),
        }.get(data_source, data_source),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Metadata saved → {path}")


def _print_comparison_hint(version_dir: Path) -> None:
    """Print existing version summaries for easy comparison."""
    versions = sorted(
        [d for d in _MODELS_ROOT.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
    )
    if len(versions) < 2:
        return
    print("\n" + "=" * 60)
    print("Model version comparison:")
    for v in versions:
        meta_path = v / "metadata.json"
        if meta_path.exists():
            try:
                m = json.loads(meta_path.read_text())
                has_model = (v / "model.pkl").exists()
                status = "✓ model.pkl" if has_model else "✗ not trained yet"
                print(f"  {v.name:<25} AP={m.get('cv_ap_mean','?'):.4f}±{m.get('cv_ap_std','?'):.4f}"
                      f"  Acc={m.get('cv_acc_mean','?'):.4f}  src={m.get('data_source','?')}"
                      f"  [{status}]")
            except Exception:
                pass
    print("=" * 60)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def _train_model_v9_hero(X: np.ndarray, y: np.ndarray):
    """
    Hero-aware HGBM with sigmoid calibration (v9 architecture).

    Designed to exploit the 9 new hero-specific features in features.py
    (`hero_call_frac`, `hero_raise_frac`, `hero_unique_amt_ratio`, etc.) which
    show Cohen's d > 3 against the v1.1 benchmark — by far the strongest
    discriminators we've measured.

    Architecture choices:
    - Single HistGradientBoostingClassifier (deeper than v7) wrapped with
      CalibratedClassifierCV(method="sigmoid"). Deeper trees so the model can
      build interactions between hero features and the structural v8 features;
      sigmoid calibration to keep outputs graded on shifted live data.
    - learning_rate=0.05 + max_iter=200 + early_stopping. Slow & steady to
      reduce overfitting (the benchmark gives perfect CV which is a warning
      sign for distribution shift).
    - max_depth=4, min_samples_leaf=40 — deeper than v7 (depth=2) to capture
      hero × all-seat interactions, but still regularized.
    - class_weight=None on the underlying HGBM (benchmark is exactly 50/50).

    Why this is expected to outperform v6/v7/v8 on live:
    - Hero-specific features are precisely "is this seat acting like a bot?"
      restricted to the labeled seat. v6-v8 were averaging the bot's signal
      with the noise from 5+ other seats per hand.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    n_samples, n_features = X.shape
    print(f"\n  [HGBM v9 hero-aware] Training on {n_samples} chunks × {n_features} features")
    print(f"  Class balance: {y.sum()} bot / {(1-y).sum()} human")

    base = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=4,
        min_samples_leaf=40,
        learning_rate=0.05,
        l2_regularization=0.5,
        max_bins=63,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(base, cv=3, method="sigmoid")),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\n  Running 5-fold cross-validation (average_precision)...")
    cv_ap  = cross_val_score(model, X, y, cv=cv, scoring="average_precision", n_jobs=1)
    cv_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy",           n_jobs=1)
    print(f"  CV AP:     {cv_ap.mean():.4f} ± {cv_ap.std():.4f}")
    print(f"  CV Acc:    {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

    cv_probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    bot_scores   = cv_probs[y == 1]
    human_scores = cv_probs[y == 0]
    print(f"\n  Score distribution (graded probabilities):")
    print(f"    Bot chunks:   mean={bot_scores.mean():.3f}  std={bot_scores.std():.3f}"
          f"  pct_above_0.5={100*np.mean(bot_scores > 0.5):.1f}%")
    print(f"    Human chunks: mean={human_scores.mean():.3f}  std={human_scores.std():.3f}"
          f"  pct_below_0.5={100*np.mean(human_scores < 0.5):.1f}%")
    ambiguous = np.mean((cv_probs > 0.3) & (cv_probs < 0.7))
    print(f"    In 0.3–0.7 range: {100*ambiguous:.1f}%")

    print("\n  Fitting final model on full dataset...")
    model.fit(X, y)

    top_features = []
    try:
        # Get feature importance from the first calibrated HGBM (averaged
        # importances across calibrated_classifiers_ folds)
        clf = model.named_steps["clf"]
        importances = np.zeros(n_features)
        n_inner = 0
        for inner_cal in clf.calibrated_classifiers_:
            inner = inner_cal.estimator
            if hasattr(inner, "feature_importances_"):
                importances += inner.feature_importances_
                n_inner += 1
        if n_inner > 0:
            importances /= n_inner
            top_idx = np.argsort(importances)[::-1][:15]
            print("\n  Top 15 features by importance:")
            for rank, idx in enumerate(top_idx, 1):
                name = CHUNK_FEATURE_NAMES[idx] if idx < len(CHUNK_FEATURE_NAMES) else f"feat_{idx}"
                marker = "  ★ HERO" if "hero_" in name else ""
                print(f"    {rank:>2}. {name:<35} {importances[idx]:.4f}{marker}")
                top_features.append({
                    "rank": rank, "name": name,
                    "importance": round(float(importances[idx]), 4),
                })
    except Exception as exc:
        print(f"  (Feature importance skipped: {exc})")

    return model, cv_ap, cv_acc, top_features


def _is_hgbm_version(version: str) -> bool:
    return (version.startswith("v3") or version.startswith("v5")
            or version.startswith("v6") or version.startswith("v7")
            or version.startswith("v8") or version.startswith("v9")
            or "gb" in version or "hgbm" in version
            or "structured" in version or "ensemble" in version
            or "benchmark" in version or "sigmoid" in version or "direct" in version
            or "hero" in version)


def main(
    version: str,
    data_source: str,
    n_chunks: int,
    human_corpus: Path,
    update_default: bool,
    seed: int,
    fast: bool,
    benchmark_path: Optional[Path] = None,
) -> None:
    n_per_class = n_chunks // 2
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    t0 = time.time()

    version_dir = _MODELS_ROOT / version
    version_dir.mkdir(parents=True, exist_ok=True)
    output_path = version_dir / "model.pkl"

    print("=" * 60)
    print(f"Poker44 BotDetector — training  [{version}]")
    print(f"  data source      : {data_source}")
    if data_source == "benchmark":
        bpath = benchmark_path or _DEFAULT_BENCHMARK_PATH
        print(f"  benchmark file   : {bpath}")
    else:
        print(f"  chunks per class : {n_per_class}")
        print(f"  chunk size range : {_CHUNK_SIZE_MIN}–{_CHUNK_SIZE_MAX} hands")
        print(f"  human corpus     : {human_corpus}")
    print(f"  output           : {output_path}")
    print(f"  update default   : {update_default}")
    print("=" * 60)

    # ---- Step 1: load / generate data ----
    if data_source == "benchmark":
        print("\n[A] Loading real labeled chunks from benchmark API download...")
        bot_chunks, human_chunks = _load_benchmark_data(benchmark_path, rng=np_rng)
        print(f"  Using ALL available benchmark chunks (no synthetic generation)")
    elif data_source == "mixed":
        bot_chunks_raw, human_chunks_raw = _generate_mixed_data(
            n_per_class, human_corpus, rng, fast=fast
        )
        # ---- Step 2 (synthetic only): sanitize ----
        print("\n[SANITIZE] Applying validator sanitization to all chunks...")
        bot_chunks   = _apply_sanitization(bot_chunks_raw)
        human_chunks = _apply_sanitization(human_chunks_raw)
        print(f"  Sanitized {len(bot_chunks)} bot + {len(human_chunks)} human chunks")
    else:
        bot_chunks_raw, human_chunks_raw = _generate_synthetic_data(
            n_per_class, human_corpus, rng
        )
        print("\n[SANITIZE] Applying validator sanitization to all chunks...")
        bot_chunks   = _apply_sanitization(bot_chunks_raw)
        human_chunks = _apply_sanitization(human_chunks_raw)
        print(f"  Sanitized {len(bot_chunks)} bot + {len(human_chunks)} human chunks")

    # ---- Step 3: features + train ----
    arch = "HistGBM" if _is_hgbm_version(version) else "RandomForest"
    print(f"\n[TRAIN] Building feature matrix + training {arch}...")
    X, y = build_training_matrix(bot_chunks, human_chunks)
    model, cv_ap, cv_acc, top_features = train_model(X, y, version=version)

    # ---- Step 4: save model ----
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    size_kb = output_path.stat().st_size // 1024
    print(f"\n  Model saved → {output_path}  ({size_kb} KB)")

    # ---- Step 5: save metadata ----
    training_seconds = time.time() - t0
    _save_metadata(
        path=version_dir / "metadata.json",
        version=version,
        data_source=data_source,
        n_per_class=min(len(bot_chunks), len(human_chunks)),
        cv_ap=cv_ap,
        cv_acc=cv_acc,
        top_features=top_features,
        training_seconds=training_seconds,
        n_features=X.shape[1],
        n_bot_chunks=len(bot_chunks),
        n_human_chunks=len(human_chunks),
    )

    # ---- Step 6: optionally update default slot ----
    if update_default:
        import shutil
        shutil.copy2(output_path, _DEFAULT_OUTPUT)
        print(f"\n  Default model updated → {_DEFAULT_OUTPUT}")

    # ---- Summary ----
    _print_comparison_hint(version_dir)
    print(f"\nTotal training time: {training_seconds / 60:.1f} minutes")
    print("=" * 60)
    print("Next steps:")
    print(f"  # A/B test: set MODEL_VERSION={version} on miners you want to upgrade")
    print(f"  # VPS: export MODEL_VERSION={version} && pm2 restart <miner_name>")
    print(f"  # Promote: python -m poker44.miner_model.train --version {version} --update-default")
    print(f"  # Push: git add poker44/miner_model/models/{version}/ && git commit && git push")
    print("=" * 60)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Poker44 bot-detection model (versioned)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--version", type=str, default="v3_gb_mixed",
        help="Model version name, saved to models/<version>/  (default: v3_gb_mixed)",
    )
    p.add_argument(
        "--data-source", choices=["synthetic", "mixed", "benchmark"], default="mixed",
        help=(
            "benchmark: use real labeled chunks from the public benchmark API (RECOMMENDED). "
            "mixed: uses build_mixed_labeled_chunks() like the validator (synthetic). "
            "synthetic: custom bot profiles, no anti-shortcut filter (legacy). "
            "Default: mixed"
        ),
    )
    p.add_argument(
        "--benchmark-path", type=Path, default=None,
        help=(
            f"Path to downloaded benchmark JSON.GZ file "
            f"(default: {_DEFAULT_BENCHMARK_PATH}). "
            "Download with: python scripts/download_benchmark.py"
        ),
    )
    p.add_argument(
        "--fast", action="store_true",
        help=f"Quick mode: {_FAST_N_CHUNKS} chunks, fewer bot candidates — faster but less accurate",
    )
    p.add_argument(
        "--n-chunks", type=int, default=None,
        help=f"Total training chunks (default {_DEFAULT_N_CHUNKS}; overridden by --fast)",
    )
    p.add_argument(
        "--human-corpus", type=Path, default=_DEFAULT_CORPUS,
        help=f"Path to human hands JSON or JSON.GZ (default: {_DEFAULT_CORPUS})",
    )
    p.add_argument(
        "--update-default", action="store_true",
        help="Also copy the trained model to model.pkl (the legacy default slot used by unversioned miners)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    n = args.n_chunks if args.n_chunks else (_FAST_N_CHUNKS if args.fast else _DEFAULT_N_CHUNKS)
    main(
        version=args.version,
        data_source=args.data_source,
        n_chunks=n,
        human_corpus=args.human_corpus,
        update_default=args.update_default,
        seed=args.seed,
        benchmark_path=getattr(args, "benchmark_path", None),
        fast=args.fast,
    )
