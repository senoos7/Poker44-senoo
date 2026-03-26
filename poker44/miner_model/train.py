"""
Training script for the Poker44 bot-detection model.

Improvements over v1:
  - 2000 labeled chunks (1000 per class, up from 300)
  - 15 diverse bot profiles covering the full tightness/aggression space
  - Random chunk sizes matching real validator range (60-120 hands)
  - RandomForestClassifier: trains 5-10x faster than GBM, comparable accuracy
  - CalibratedClassifierCV for better probability scores (AP reward)
  - Feature importance report printed after training
  - Model saved to git-tracked poker44/miner_model/model.pkl for push → pull workflow

Usage:
    cd /path/to/Poker44-subnet
    python -m poker44.miner_model.train           # default (2000 chunks)
    python -m poker44.miner_model.train --fast    # 800 chunks, quicker iteration
    python -m poker44.miner_model.train --n-chunks 3000  # larger model
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker44.miner_model.features import extract_chunk_features, CHUNK_FEATURE_NAMES
from poker44.miner_model.sanitize import sanitize_hand

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
_DEFAULT_CORPUS       = REPO_ROOT / "hands_generator" / "human_hands" / "poker_hands_combined.json.gz"
_DEFAULT_OUTPUT       = Path(__file__).parent / "model.pkl"
_DEFAULT_N_CHUNKS     = 2000   # total labeled chunks (half bot, half human)
_CHUNK_SIZE_MIN       = 60     # matches real validator HANDS_PER_CHUNK_RANGE
_CHUNK_SIZE_MAX       = 120


# ------------------------------------------------------------------
# Extended bot profiles — covers the full playing-style space
# ------------------------------------------------------------------

def _extended_bot_profiles():
    from hands_generator.bot_hands.generate_poker_data import BotProfile
    return [
        # --- Original 5 ---
        BotProfile(name="balanced",         tightness=0.54, aggression=0.66, bluff_freq=0.05),
        BotProfile(name="tight_aggressive", tightness=0.62, aggression=0.80, bluff_freq=0.04),
        BotProfile(name="loose_aggressive", tightness=0.44, aggression=0.78, bluff_freq=0.08),
        BotProfile(name="tight_passive",    tightness=0.60, aggression=0.50, bluff_freq=0.02),
        BotProfile(name="loose_passive",    tightness=0.46, aggression=0.46, bluff_freq=0.05),
        # --- Extended: edge-case styles ---
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
# Data generation helpers
# ------------------------------------------------------------------

def _load_human_corpus(path: Path) -> List[Dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        hands = json.load(f)
    print(f"  Loaded {len(hands):,} human hands from {path.name}")
    return hands


def _generate_bot_chunks(
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


def _apply_sanitization(chunks: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
    return [[sanitize_hand(h) for h in chunk] for chunk in chunks]


# ------------------------------------------------------------------
# Feature matrix builder
# ------------------------------------------------------------------

def build_training_matrix(
    bot_chunks: List[List[Dict[str, Any]]],
    human_chunks: List[List[Dict[str, Any]]],
) -> tuple[np.ndarray, np.ndarray]:
    all_chunks = bot_chunks + human_chunks
    labels = np.array([1] * len(bot_chunks) + [0] * len(human_chunks), dtype=int)
    X = np.vstack([extract_chunk_features(c) for c in all_chunks])
    idx = np.random.default_rng(42).permutation(len(labels))
    return X[idx], labels[idx]


# ------------------------------------------------------------------
# Model training
# ------------------------------------------------------------------

def train_model(X: np.ndarray, y: np.ndarray):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    n_samples, n_features = X.shape
    print(f"\n  Training on {n_samples} chunks × {n_features} features")
    print(f"  Class balance: {y.sum()} bot / {(1-y).sum()} human")

    # RandomForest: much faster than GBM, still very accurate
    # n_estimators=400 gives good stability; min_samples_leaf avoids overfit
    base = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,          # use all CPU cores
        random_state=42,
    )
    # Isotonic calibration improves AP score (reward weights AP at 65%)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(base, cv=3, method="isotonic")),
    ])

    # 5-fold CV to estimate real performance before final fit
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\n  Running 5-fold cross-validation (average_precision)...")
    cv_ap = cross_val_score(model, X, y, cv=cv, scoring="average_precision", n_jobs=1)
    print(f"  CV AP:     {cv_ap.mean():.4f} ± {cv_ap.std():.4f}  (target >0.80)")

    cv_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    print(f"  CV Acc:    {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

    # Fit final model on all data
    print("\n  Fitting final model on full dataset...")
    model.fit(X, y)

    # Feature importance from the inner RF
    rf = model.named_steps["clf"].calibrated_classifiers_[0].estimator
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    print("\n  Top 10 features by importance:")
    for rank, idx in enumerate(top_idx, 1):
        feat_name = CHUNK_FEATURE_NAMES[idx] if idx < len(CHUNK_FEATURE_NAMES) else f"feat_{idx}"
        print(f"    {rank:>2}. {feat_name:<35} {importances[idx]:.4f}")

    # --- CRITICAL: set n_jobs=1 on all inner RandomForests before saving ---
    # Training used n_jobs=-1 (all cores) for speed.
    # Inference runs inside an async miner coroutine — joblib parallel workers
    # cause "delayed should be used with Parallel" warnings and severe slowdowns.
    # Setting n_jobs=1 here makes predict_proba() fully synchronous at inference time.
    clf = model.named_steps["clf"]
    for cal_clf in clf.calibrated_classifiers_:
        if hasattr(cal_clf, "estimator"):
            cal_clf.estimator.n_jobs = 1
        if hasattr(cal_clf, "base_estimator"):       # sklearn < 1.2 compat
            cal_clf.base_estimator.n_jobs = 1
    print("\n  Set n_jobs=1 on all inner RandomForests (safe for async inference).")

    return model


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(
    n_chunks: int = _DEFAULT_N_CHUNKS,
    human_corpus: Path = _DEFAULT_CORPUS,
    output_path: Path = _DEFAULT_OUTPUT,
    seed: int = 42,
) -> None:
    n_per_class = n_chunks // 2
    rng = random.Random(seed)

    print("=" * 60)
    print("Poker44 BotDetector — training (v2)")
    print(f"  chunks per class : {n_per_class}")
    print(f"  chunk size range : {_CHUNK_SIZE_MIN}–{_CHUNK_SIZE_MAX} hands")
    print(f"  bot profiles     : {len(_extended_bot_profiles())}")
    print(f"  human corpus     : {human_corpus}")
    print(f"  output           : {output_path}")
    print("=" * 60)

    print("\n[1/5] Loading human corpus...")
    human_hands = _load_human_corpus(human_corpus)

    print(f"\n[2/5] Generating {n_per_class} bot chunks ({_CHUNK_SIZE_MIN}–{_CHUNK_SIZE_MAX} hands each)...")
    bot_chunks_raw = _generate_bot_chunks(n_per_class, rng)

    print(f"\n[3/5] Sampling {n_per_class} human chunks...")
    human_chunks_raw = _sample_human_chunks(human_hands, n_per_class, rng)

    print("\n[4/5] Applying validator sanitization...")
    bot_chunks   = _apply_sanitization(bot_chunks_raw)
    human_chunks = _apply_sanitization(human_chunks_raw)

    print("\n[5/5] Building features + training RandomForest...")
    X, y = build_training_matrix(bot_chunks, human_chunks)
    model = train_model(X, y)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    size_kb = output_path.stat().st_size // 1024
    print(f"\n  Model saved → {output_path}  ({size_kb} KB)")
    print("=" * 60)
    print("Done. Next steps:")
    print("  git add poker44/miner_model/model.pkl && git commit -m 'retrain model v2' && git push")
    print("  # On VPS: git pull && ./scripts/miner/run/run_multi_miner.sh stop && start")
    print("=" * 60)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Poker44 bot-detection model")
    p.add_argument("--fast",       action="store_true",
                   help="Quick mode: 800 chunks for faster iteration")
    p.add_argument("--n-chunks",   type=int, default=None,
                   help=f"Total training chunks (default {_DEFAULT_N_CHUNKS})")
    p.add_argument("--human-corpus", type=Path, default=_DEFAULT_CORPUS)
    p.add_argument("--output",     type=Path, default=_DEFAULT_OUTPUT)
    p.add_argument("--seed",       type=int,  default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    n = args.n_chunks if args.n_chunks else (800 if args.fast else _DEFAULT_N_CHUNKS)
    main(
        n_chunks=n,
        human_corpus=args.human_corpus,
        output_path=args.output,
        seed=args.seed,
    )
