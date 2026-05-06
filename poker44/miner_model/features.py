"""
Feature extraction for Poker44 bot detection.

Works on the REAL sanitized hand format from poker44/validator/payload_view.py:
  - actions: exactly 12 sampled slots, each with:
      street, action_type, normalized_amount_bb, pot_before, pot_after,
      actor_seat, amount, raise_to, call_to
  - outcome: total_pot=0.0, rake=0.0 (always — do NOT use these)
  - players: active seats with starting_stack in standardized BB units
  - streets: [] (community cards stripped)
  - metadata: normalized to constants (sb=0.01, bb=0.02)

Per-hand feature vector (25 features):
  preflop_frac, flop_frac, turn_frac, river_frac,   [street distribution]
  depth,                                             [postflop depth 0-1]
  fold_frac, call_frac, raise_frac, check_frac,      [action type fracs]
  amount_mean, amount_std,                           [bet sizing in BB]
  pot_after_last,                                    [final pot in BB]
  n_players, stack_mean, stack_cv,                   [player/stack info]
  aggression,                                        [raise / (call+check)]
  bet_cv,                                            [std/mean of bet sizes]
  went_to_river,                                     [1.0 if river seen]
  street_entropy,                                    [entropy of street dist]
  unique_bet_ratio,                                  [unique bet sizes / non-zero bets — bots reuse sizes]
  preflop_raise_frac,                                [raises as fraction of preflop actions only]
  max_amount_bb,                                     [max single bet in BB, capped at 50]
  blind_frac,                                        [fraction of actions that are just blinds]
  call_raise_ratio,                                  [calls / raises — passivity signal]
  n_actions_norm,                                    [total actions / window (12) — hand length proxy]

Chunk-level feature vector (100 features):
  mean, std, p25, p75 of the 25 per-hand features across all hands in the chunk.

Key discriminators:
  - LOW within-chunk variance → systematic bot profiles
  - LOW unique_bet_ratio → mechanical bet sizing (always 2BB, 3BB, etc.)
  - LOW street_entropy → bots concentrate on fewer streets (more preflop folds)
  - HIGH blind_frac → bots fold fast, window dominated by blind postings
  - LOW bet_cv → mechanical, non-adaptive sizing
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List

import numpy as np

_MINER_ACTION_WINDOW = 12
_POST_FLOP_STREETS = {"flop", "turn", "river"}
_N_HAND_FEATURES = 25

_RAISE_TYPES = {"raise", "bet", "all_in"}
_CALL_TYPES  = {"call"}
_CHECK_TYPES = {"check"}
_FOLD_TYPES  = {"fold"}
_BLIND_TYPES = {"small_blind", "big_blind", "ante"}

_LOG4 = math.log(4.0)   # max possible street entropy denominator


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _safe_str(v: Any) -> str:
    return str(v).strip().lower() if v is not None else ""


def extract_hand_features(hand: Dict[str, Any]) -> np.ndarray:
    """Return a (25,) float32 feature vector for one sanitized hand."""
    actions = hand.get("actions") or []
    players = hand.get("players") or []

    total_slots = max(len(actions), 1)

    # Pre-compute strings and amounts ONCE per action to avoid redundant
    # str().strip().lower() calls (major hotspot at 2400 hands per query).
    act_types = [str(a.get("action_type") or "").strip().lower() for a in actions]
    act_streets = [str(a.get("street") or "").strip().lower() for a in actions]
    act_amounts_raw = [a.get("normalized_amount_bb") for a in actions]
    act_amounts = [
        float(v) if v is not None else 0.0
        for v in act_amounts_raw
    ]
    act_pot_after = [
        float(a.get("pot_after") or 0.0) for a in actions
    ]

    # --- Street distribution ---
    street_counts = Counter(act_streets)
    preflop_frac = street_counts.get("preflop", 0) / total_slots
    flop_frac    = street_counts.get("flop",    0) / total_slots
    turn_frac    = street_counts.get("turn",    0) / total_slots
    river_frac   = street_counts.get("river",   0) / total_slots

    distinct_postflop = sum(1 for s in _POST_FLOP_STREETS if street_counts.get(s, 0) > 0)
    depth = distinct_postflop / 3.0

    # --- Action type fractions ---
    type_counts = Counter(act_types)
    fold_frac  = sum(type_counts.get(t, 0) for t in _FOLD_TYPES)  / total_slots
    call_frac  = sum(type_counts.get(t, 0) for t in _CALL_TYPES)  / total_slots
    raise_frac = sum(type_counts.get(t, 0) for t in _RAISE_TYPES) / total_slots
    check_frac = sum(type_counts.get(t, 0) for t in _CHECK_TYPES) / total_slots

    # --- Bet sizing: non-zero normalized_amount_bb ---
    non_zero_amounts = [v for v in act_amounts if v > 0.0]
    amount_mean = sum(non_zero_amounts) / len(non_zero_amounts) if non_zero_amounts else 0.0
    if len(non_zero_amounts) > 1:
        _m = amount_mean
        amount_std = (sum((v - _m) ** 2 for v in non_zero_amounts) / len(non_zero_amounts)) ** 0.5
    else:
        amount_std = 0.0

    # --- Final pot size (last action's pot_after, normalized by BB=0.02) ---
    pot_after_last = min(act_pot_after[-1] / 0.02, 500.0) if act_pot_after else 0.0

    # --- Stack features ---
    stacks = [float(p.get("starting_stack") or 0.0) for p in players
              if (p.get("starting_stack") or 0.0) > 0.0]
    n_players  = len(stacks) / 6.0
    if stacks:
        stack_s = sum(stacks)
        stack_mean = (stack_s / len(stacks)) / 2.0  # normalise ~100BB → ~1.0
        if len(stacks) > 1:
            _sm = stack_s / len(stacks)
            stack_std = (sum((v - _sm) ** 2 for v in stacks) / len(stacks)) ** 0.5
            stack_cv  = stack_std / max(_sm, 1e-6)
        else:
            stack_std = 0.0
            stack_cv  = 0.0
    else:
        stack_mean = stack_cv = 0.0

    # --- Aggression ratio ---
    aggression = min(raise_frac / max(call_frac + check_frac, 1e-6), 5.0)

    # --- Bet coefficient of variation ---
    bet_cv = min(amount_std / max(amount_mean, 1e-6), 5.0)

    # --- River flag ---
    went_to_river = 1.0 if street_counts.get("river", 0) > 0 else 0.0

    # --- Street entropy ---
    fracs = [preflop_frac, flop_frac, turn_frac, river_frac]
    raw_entropy = -sum(f * math.log(f + 1e-9) for f in fracs)
    street_entropy = min(max(raw_entropy / _LOG4, 0.0), 1.0)

    # --- Unique bet ratio ---
    if non_zero_amounts:
        unique_bets = len(set(round(x, 2) for x in non_zero_amounts))
        unique_bet_ratio = unique_bets / len(non_zero_amounts)
    else:
        unique_bet_ratio = 0.0

    # --- Preflop raise fraction ---
    pf_n = sum(1 for s in act_streets if s == "preflop")
    pf_raises = sum(1 for t, s in zip(act_types, act_streets)
                    if s == "preflop" and t in _RAISE_TYPES)
    preflop_raise_frac = pf_raises / max(pf_n, 1)

    # --- Max single bet size (capped at 50 BB) ---
    max_amount_bb = min(max(non_zero_amounts, default=0.0), 50.0)

    # --- Blind fraction ---
    blind_count = sum(1 for t in act_types if t in _BLIND_TYPES)
    blind_frac = blind_count / total_slots

    # --- Call-to-raise ratio ---
    n_raises = sum(type_counts.get(t, 0) for t in _RAISE_TYPES)
    call_raise_ratio = min(type_counts.get("call", 0) / max(n_raises, 1), 5.0)

    # --- Normalized action count ---
    n_actions_norm = len(actions) / _MINER_ACTION_WINDOW

    return np.array(
        [
            preflop_frac, flop_frac, turn_frac, river_frac,
            depth,
            fold_frac, call_frac, raise_frac, check_frac,
            amount_mean, amount_std,
            pot_after_last,
            n_players, stack_mean, stack_cv,
            aggression,
            bet_cv,
            went_to_river,
            street_entropy,
            unique_bet_ratio,
            preflop_raise_frac,
            max_amount_bb,
            blind_frac,
            call_raise_ratio,
            n_actions_norm,
        ],
        dtype=np.float32,
    )


def extract_chunk_features(chunk: List[Dict[str, Any]]) -> np.ndarray:
    """
    Return a (76,) float32 feature vector for one chunk of sanitized hands.

    Aggregates per-hand features with mean, std, p25, p75.
    The std components are the strongest bot-vs-human discriminators:
    bot chunks have LOW variance (systematic profiles), humans have HIGH variance.
    """
    if not chunk:
        return np.zeros(4 * _N_HAND_FEATURES, dtype=np.float32)

    hand_mat = np.vstack([extract_hand_features(h) for h in chunk])  # (N, 19)

    means = hand_mat.mean(axis=0)
    stds  = hand_mat.std(axis=0)
    p25   = np.percentile(hand_mat, 25, axis=0)
    p75   = np.percentile(hand_mat, 75, axis=0)

    return np.concatenate([means, stds, p25, p75]).astype(np.float32)


_FEAT_NAMES = [
    "preflop_frac", "flop_frac", "turn_frac", "river_frac",
    "depth",
    "fold_frac", "call_frac", "raise_frac", "check_frac",
    "amount_mean", "amount_std",
    "pot_after_last",
    "n_players", "stack_mean", "stack_cv",
    "aggression",
    "bet_cv",
    "went_to_river",
    "street_entropy",
    "unique_bet_ratio",
    "preflop_raise_frac",
    "max_amount_bb",
    "blind_frac",
    "call_raise_ratio",
    "n_actions_norm",
]

assert len(_FEAT_NAMES) == _N_HAND_FEATURES, (
    f"_FEAT_NAMES has {len(_FEAT_NAMES)} entries but _N_HAND_FEATURES={_N_HAND_FEATURES}"
)

CHUNK_FEATURE_NAMES: List[str] = [
    f"{stat}_{feat}"
    for stat in ("mean", "std", "p25", "p75")
    for feat in _FEAT_NAMES
]
