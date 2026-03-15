"""
Feature extraction for Poker44 bot detection.

Works on the SANITIZED hand format miners receive from validators:
  - actions: 12 slots, each with only `street` field meaningful (action_type="action", amounts=0)
  - outcome: total_pot, rake (showdown/winners stripped)
  - players: 6 fixed seats, starting_stack only (hole_cards=None, uid anonymised)
  - streets: [] (stripped)
  - metadata: sb/bb/seats normalised to constants

Per-hand feature vector (10 features):
  preflop_frac, flop_frac, turn_frac, river_frac,
  depth (0-3 post-flop streets reached), total_pot, rake,
  n_players, stack_mean_norm, stack_cv

Chunk-level feature vector (40 features):
  mean, std, p25, p75 of the per-hand vector across all hands in the chunk.

The key discriminator is LOW STD inside a bot chunk (systematic bot behaviour)
vs HIGH STD inside a human chunk (diverse human playing styles).
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import numpy as np

# ------------------------------------------------------------------
# Constants matching forward.py sanitization
# ------------------------------------------------------------------
_MINER_ACTION_WINDOW = 12
_STREETS_ORDER = ["preflop", "flop", "turn", "river"]
_POST_FLOP_STREETS = {"flop", "turn", "river"}
_N_HAND_FEATURES = 10


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------
# Per-hand feature extraction
# ------------------------------------------------------------------

def extract_hand_features(hand: Dict[str, Any]) -> np.ndarray:
    """Return a (10,) float32 feature vector for one sanitized hand."""
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    outcome = hand.get("outcome") or {}

    # --- Street distribution (each of 12 slots has a street label) ---
    street_counts = Counter(_safe_str(a.get("street", "")) for a in actions)
    total_slots = max(len(actions), 1)

    preflop_frac = street_counts.get("preflop", 0) / total_slots
    flop_frac    = street_counts.get("flop",    0) / total_slots
    turn_frac    = street_counts.get("turn",    0) / total_slots
    river_frac   = street_counts.get("river",   0) / total_slots

    # Depth: how many distinct post-flop streets appear (0-3)
    distinct_postflop = sum(1 for s in _POST_FLOP_STREETS if street_counts.get(s, 0) > 0)
    depth = distinct_postflop / 3.0  # normalise to [0,1]

    # --- Outcome ---
    total_pot = _safe_float(outcome.get("total_pot"))
    rake      = _safe_float(outcome.get("rake"))

    # --- Players ---
    stacks = [
        _safe_float(p.get("starting_stack"))
        for p in players
        if _safe_float(p.get("starting_stack")) > 0.0
    ]
    n_players = len(stacks) / 6.0  # normalise to [0,1]
    if stacks:
        stack_mean = float(np.mean(stacks))
        stack_std  = float(np.std(stacks))
        stack_cv   = stack_std / stack_mean if stack_mean > 0 else 0.0
    else:
        stack_mean = 0.0
        stack_cv   = 0.0

    return np.array(
        [preflop_frac, flop_frac, turn_frac, river_frac,
         depth, total_pot, rake, n_players, stack_mean, stack_cv],
        dtype=np.float32,
    )


def _safe_str(v: Any) -> str:
    return str(v) if v is not None else ""


# ------------------------------------------------------------------
# Chunk-level feature extraction
# ------------------------------------------------------------------

def extract_chunk_features(chunk: List[Dict[str, Any]]) -> np.ndarray:
    """
    Return a (40,) float32 feature vector for one chunk of sanitized hands.

    Aggregates per-hand features with mean, std, p25, p75.
    The std components are the strongest bot-vs-human discriminators because
    bot chunks (same profile family) have LOW variance while human chunks are
    highly diverse.
    """
    if not chunk:
        return np.zeros(4 * _N_HAND_FEATURES, dtype=np.float32)

    hand_mat = np.vstack([extract_hand_features(h) for h in chunk])  # (N, 10)

    means = hand_mat.mean(axis=0)
    stds  = hand_mat.std(axis=0)
    p25   = np.percentile(hand_mat, 25, axis=0)
    p75   = np.percentile(hand_mat, 75, axis=0)

    return np.concatenate([means, stds, p25, p75]).astype(np.float32)


CHUNK_FEATURE_NAMES: List[str] = []
for _stat in ("mean", "std", "p25", "p75"):
    for _feat in (
        "preflop_frac", "flop_frac", "turn_frac", "river_frac",
        "depth", "total_pot", "rake", "n_players", "stack_mean", "stack_cv",
    ):
        CHUNK_FEATURE_NAMES.append(f"{_stat}_{_feat}")
