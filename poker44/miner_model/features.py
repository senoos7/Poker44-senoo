"""
Feature extraction for Poker44 bot detection.

Works on the REAL sanitized hand format from poker44/validator/sanitization.py:
  - actions: exactly 12 sampled slots, each with:
      street, action_type, normalized_amount_bb, pot_before, pot_after,
      actor_seat, amount, raise_to, call_to
  - outcome: total_pot=0.0, rake=0.0 (always — do NOT use these)
  - players: active seats with starting_stack in standardized BB units
  - streets: [] (community cards stripped)
  - metadata: normalized to constants (sb=0.01, bb=0.02)

Per-hand feature vector (15 features):
  preflop_frac, flop_frac, turn_frac, river_frac,   [street distribution]
  depth,                                             [postflop depth 0-1]
  fold_frac, call_frac, raise_frac,                  [action type fracs]
  amount_mean, amount_std,                           [bet sizing in BB]
  pot_after_last,                                    [final pot in BB]
  n_players, stack_mean, stack_cv                    [player/stack info]

Chunk-level feature vector (60 features):
  mean, std, p25, p75 of the per-hand vector across all hands in the chunk.

Key discriminator: bots play with LOW within-chunk variance (systematic profiles)
while human chunks are diverse — high variance in bet sizes, street depth, etc.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import numpy as np

_MINER_ACTION_WINDOW = 12
_POST_FLOP_STREETS = {"flop", "turn", "river"}
_N_HAND_FEATURES = 15

_RAISE_TYPES = {"raise", "bet", "all_in"}
_CALL_TYPES  = {"call"}
_CHECK_TYPES = {"check"}
_FOLD_TYPES  = {"fold"}


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _safe_str(v: Any) -> str:
    return str(v).strip().lower() if v is not None else ""


def extract_hand_features(hand: Dict[str, Any]) -> np.ndarray:
    """Return a (15,) float32 feature vector for one sanitized hand."""
    actions = hand.get("actions") or []
    players = hand.get("players") or []

    total_slots = max(len(actions), 1)

    # --- Street distribution ---
    street_counts = Counter(_safe_str(a.get("street", "")) for a in actions)
    preflop_frac = street_counts.get("preflop", 0) / total_slots
    flop_frac    = street_counts.get("flop",    0) / total_slots
    turn_frac    = street_counts.get("turn",    0) / total_slots
    river_frac   = street_counts.get("river",   0) / total_slots

    distinct_postflop = sum(1 for s in _POST_FLOP_STREETS if street_counts.get(s, 0) > 0)
    depth = distinct_postflop / 3.0

    # --- Action type fractions ---
    type_counts = Counter(_safe_str(a.get("action_type", "")) for a in actions)
    fold_frac  = sum(type_counts.get(t, 0) for t in _FOLD_TYPES)  / total_slots
    call_frac  = sum(type_counts.get(t, 0) for t in _CALL_TYPES)  / total_slots
    raise_frac = sum(type_counts.get(t, 0) for t in _RAISE_TYPES) / total_slots

    # --- Bet sizing: normalized_amount_bb across active (non-zero) actions ---
    amounts = [
        _safe_float(a.get("normalized_amount_bb"))
        for a in actions
        if _safe_float(a.get("normalized_amount_bb")) > 0.0
    ]
    amount_mean = float(np.mean(amounts)) if amounts else 0.0
    amount_std  = float(np.std(amounts))  if len(amounts) > 1 else 0.0

    # --- Final pot size (last action's pot_after, normalized by BB=0.02) ---
    pot_after_values = [_safe_float(a.get("pot_after")) for a in actions]
    pot_after_last = pot_after_values[-1] / 0.02 if pot_after_values else 0.0
    pot_after_last = min(pot_after_last, 500.0)  # cap at 500 BB

    # --- Stack features ---
    stacks = [
        _safe_float(p.get("starting_stack"))
        for p in players
        if _safe_float(p.get("starting_stack")) > 0.0
    ]
    n_players  = len(stacks) / 6.0
    stack_mean = float(np.mean(stacks)) / 2.0 if stacks else 0.0  # normalise ~100BB → ~1.0
    stack_std  = float(np.std(stacks)) if len(stacks) > 1 else 0.0
    stack_cv   = stack_std / max(float(np.mean(stacks)), 1e-6) if stacks else 0.0

    return np.array(
        [
            preflop_frac, flop_frac, turn_frac, river_frac,
            depth,
            fold_frac, call_frac, raise_frac,
            amount_mean, amount_std,
            pot_after_last,
            n_players, stack_mean, stack_cv,
            # 15th feature: aggression ratio (raises per call+check, capped at 5)
            min(raise_frac / max(call_frac + (type_counts.get("check", 0) / total_slots), 1e-6), 5.0),
        ],
        dtype=np.float32,
    )


def extract_chunk_features(chunk: List[Dict[str, Any]]) -> np.ndarray:
    """
    Return a (60,) float32 feature vector for one chunk of sanitized hands.

    Aggregates per-hand features with mean, std, p25, p75.
    The std components are the strongest bot-vs-human discriminators:
    bot chunks have LOW variance (systematic profiles), humans have HIGH variance.
    """
    if not chunk:
        return np.zeros(4 * _N_HAND_FEATURES, dtype=np.float32)

    hand_mat = np.vstack([extract_hand_features(h) for h in chunk])  # (N, 15)

    means = hand_mat.mean(axis=0)
    stds  = hand_mat.std(axis=0)
    p25   = np.percentile(hand_mat, 25, axis=0)
    p75   = np.percentile(hand_mat, 75, axis=0)

    return np.concatenate([means, stds, p25, p75]).astype(np.float32)


_FEAT_NAMES = [
    "preflop_frac", "flop_frac", "turn_frac", "river_frac",
    "depth",
    "fold_frac", "call_frac", "raise_frac",
    "amount_mean", "amount_std",
    "pot_after_last",
    "n_players", "stack_mean", "stack_cv",
    "aggression",
]

CHUNK_FEATURE_NAMES: List[str] = [
    f"{stat}_{feat}"
    for stat in ("mean", "std", "p25", "p75")
    for feat in _FEAT_NAMES
]
