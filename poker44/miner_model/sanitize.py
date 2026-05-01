"""
Validator-equivalent sanitization for training data.

Delegates directly to prepare_hand_for_miner() in
poker44/validator/payload_view.py so training data is byte-for-byte
identical to what miners receive at inference time.

Previously this file had its own copy that differed in two critical ways:
  1. It preserved total_pot/rake — the real sanitizer always sets them to 0.0
  2. It zeroed action_type/amounts — the real sanitizer keeps them
Both caused a large distribution shift that made the model output 1.0 for everything.

Note: upstream renamed sanitization.py → payload_view.py and
sanitize_hand_for_miner → prepare_hand_for_miner (v1 refactor, May 2026).
"""

from __future__ import annotations

from typing import Any, Dict

from poker44.validator.payload_view import prepare_hand_for_miner


def sanitize_hand(hand_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize a raw hand dict exactly as the validator does."""
    return prepare_hand_for_miner(hand_payload)
