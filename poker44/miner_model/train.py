"""
Training script for the Poker44 bot-detection model.

Generates labeled bot + human chunks, applies the same sanitization the
validator applies, extracts features, and trains a GradientBoostingClassifier.
The model is saved to poker44/miner_model/model.pkl.

Usage:
    cd /path/to/Poker44-subnet
    python -m poker44.miner_model.train
    # or with options:
    python -m poker44.miner_model.train \\
        --n-chunks 600 \\
        --hands-per-chunk 80 \\
        --human-corpus hands_generator/human_hands/poker_hands_combined.json.gz \\
        --output poker44/miner_model/model.pkl

Training data:
    - Bot chunks:   generated live by hands_generator (same code as validator)
    - Human chunks: sampled from public corpus (or private if provided)
    - Both passed through validator-equivalent sanitization so the model
      trains on exactly what miners receive at inference time.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker44.miner_model.features import extract_chunk_features
from poker44.miner_model.sanitize import sanitize_hand

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
_DEFAULT_CORPUS = REPO_ROOT / "hands_generator" / "human_hands" / "poker_hands_combined.json.gz"
_DEFAULT_OUTPUT = Path(__file__).parent / "model.pkl"
_DEFAULT_N_CHUNKS = 600        # total labeled chunks (half bot, half human)
_DEFAULT_HANDS_PER_CHUNK = 80  # hands per chunk during training


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
    hands_per_chunk: int,
    seed: int,
) -> List[List[Dict[str, Any]]]:
    from hands_generator.data_generator import generate_bot_chunk, _default_bot_profiles
    profiles = _default_bot_profiles()
    rng = random.Random(seed)
    chunks = []
    for i in range(n_chunks):
        chunk = generate_bot_chunk(
            size=hands_per_chunk,
            profiles=profiles,
            seed=rng.randint(0, 10**9),
        )
        chunks.append(chunk)
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{n_chunks} bot chunks...")
    return chunks


def _sample_human_chunks(
    human_hands: List[Dict[str, Any]],
    n_chunks: int,
    hands_per_chunk: int,
    seed: int,
) -> List[List[Dict[str, Any]]]:
    rng = random.Random(seed + 1)
    chunks = []
    for _ in range(n_chunks):
        size = min(hands_per_chunk, len(human_hands))
        chunks.append(rng.sample(human_hands, size))
    return chunks


def _apply_sanitization(chunks: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
    """Apply validator-equivalent sanitization to every hand in every chunk."""
    sanitized = []
    for chunk in chunks:
        sanitized.append([sanitize_hand(h) for h in chunk])
    return sanitized


# ------------------------------------------------------------------
# Feature + label matrix builder
# ------------------------------------------------------------------

def build_training_matrix(
    bot_chunks: List[List[Dict[str, Any]]],
    human_chunks: List[List[Dict[str, Any]]],
) -> tuple[np.ndarray, np.ndarray]:
    all_chunks = bot_chunks + human_chunks
    labels = np.array([1] * len(bot_chunks) + [0] * len(human_chunks), dtype=int)

    feature_list = [extract_chunk_features(chunk) for chunk in all_chunks]
    X = np.vstack(feature_list)

    # Shuffle to avoid ordering bias
    idx = np.random.default_rng(42).permutation(len(labels))
    return X[idx], labels[idx]


# ------------------------------------------------------------------
# Model training
# ------------------------------------------------------------------

def train_model(X: np.ndarray, y: np.ndarray):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    print(f"\n  Training GradientBoostingClassifier on {X.shape[0]} samples, {X.shape[1]} features...")

    base = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
    )
    # Isotonic calibration improves AP (reward weights AP heavily)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(base, cv=3, method="isotonic")),
    ])

    # Cross-validation to estimate performance
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="average_precision")
    print(f"  CV average precision: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final fit on all data
    model.fit(X, y)
    return model


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(
    n_chunks: int = _DEFAULT_N_CHUNKS,
    hands_per_chunk: int = _DEFAULT_HANDS_PER_CHUNK,
    human_corpus: Path = _DEFAULT_CORPUS,
    output_path: Path = _DEFAULT_OUTPUT,
    seed: int = 42,
) -> None:
    print("=" * 60)
    print("Poker44 BotDetector — training")
    print(f"  n_chunks per class : {n_chunks // 2}")
    print(f"  hands per chunk    : {hands_per_chunk}")
    print(f"  human corpus       : {human_corpus}")
    print("=" * 60)

    # 1. Load human corpus
    print("\n[1/5] Loading human corpus...")
    human_hands = _load_human_corpus(human_corpus)

    n_per_class = n_chunks // 2

    # 2. Generate bot chunks
    print(f"\n[2/5] Generating {n_per_class} bot chunks...")
    bot_chunks_raw = _generate_bot_chunks(n_per_class, hands_per_chunk, seed)

    # 3. Sample human chunks
    print(f"\n[3/5] Sampling {n_per_class} human chunks...")
    human_chunks_raw = _sample_human_chunks(human_hands, n_per_class, hands_per_chunk, seed)

    # 4. Sanitize (simulate validator stripping)
    print("\n[4/5] Applying validator sanitization...")
    bot_chunks   = _apply_sanitization(bot_chunks_raw)
    human_chunks = _apply_sanitization(human_chunks_raw)

    # 5. Build features + train
    print("\n[5/5] Building feature matrix and training model...")
    X, y = build_training_matrix(bot_chunks, human_chunks)
    model = train_model(X, y)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n  Model saved to {output_path}")
    print("=" * 60)
    print("Training complete. Restart your miner to load the new model.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Poker44 bot-detection model")
    p.add_argument("--n-chunks", type=int, default=_DEFAULT_N_CHUNKS,
                   help=f"Total training chunks, default {_DEFAULT_N_CHUNKS}")
    p.add_argument("--hands-per-chunk", type=int, default=_DEFAULT_HANDS_PER_CHUNK,
                   help=f"Hands per chunk, default {_DEFAULT_HANDS_PER_CHUNK}")
    p.add_argument("--human-corpus", type=Path, default=_DEFAULT_CORPUS,
                   help="Path to human hands JSON or .json.gz")
    p.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT,
                   help="Output path for model.pkl")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        n_chunks=args.n_chunks,
        hands_per_chunk=args.hands_per_chunk,
        human_corpus=args.human_corpus,
        output_path=args.output,
        seed=args.seed,
    )
