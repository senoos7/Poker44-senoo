"""
Validator-equivalent sanitization for training data.

Delegates directly to the real sanitize_hand_for_miner() in
poker44/validator/sanitization.py so training data is byte-for-byte
identical to what miners receive at inference time.

Previously this file had its own copy that differed in two critical ways:
  1. It preserved total_pot/rake — the real sanitizer always sets them to 0.0
  2. It zeroed action_type/amounts — the real sanitizer keeps them
Both caused a large distribution shift that made the model output 1.0 for everything.
"""

from __future__ import annotations

from typing import Any, Dict

from poker44.validator.sanitization import sanitize_hand_for_miner


def sanitize_hand(hand_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize a raw hand dict exactly as the validator does."""
    return sanitize_hand_for_miner(hand_payload)
