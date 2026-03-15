"""
Validator-equivalent sanitization for training data.

Mirrors _sanitize_hand_for_miner() in poker44/validator/forward.py exactly
so that training samples match what miners receive at inference time.

Constants must be kept in sync with forward.py:
  _MINER_ACTION_WINDOW = 12
  _DEFAULT_MAX_SEATS   = 6
  _SANITIZED_SB/BB/ANTE
"""

from __future__ import annotations

from typing import Any, Dict, List

_LEAKAGE_KEYS = {"label", "label_flag", "is_bot", "bot_family_id", "bot_version"}
_MINER_ACTION_WINDOW = 12
_DEFAULT_MAX_SEATS = 6
_SANITIZED_SB   = 0.01
_SANITIZED_BB   = 0.02
_SANITIZED_ANTE = 0.0


def _strip_leakage(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_leakage(v) for k, v in value.items() if k not in _LEAKAGE_KEYS}
    if isinstance(value, list):
        return [_strip_leakage(item) for item in value]
    return value


def sanitize_hand(hand_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove ground-truth leakage — mirrors forward.py _sanitize_hand_for_miner."""
    cleaned = _strip_leakage(hand_payload)
    if not isinstance(cleaned, dict):
        return {}

    metadata   = cleaned.get("metadata") if isinstance(cleaned.get("metadata"), dict) else {}
    players_raw = cleaned.get("players") if isinstance(cleaned.get("players"), list) else []
    actions_raw = cleaned.get("actions") if isinstance(cleaned.get("actions"), list) else []
    outcome     = cleaned.get("outcome")

    # --- Players: fixed 6-seat grid, anonymise uid, keep stack only ---
    seat_to_stack: Dict[int, float] = {}
    for p in players_raw:
        if not isinstance(p, dict):
            continue
        try:
            seat_i = int(p.get("seat"))
        except (TypeError, ValueError):
            continue
        if seat_i <= 0:
            continue
        seat_to_stack[seat_i] = float(p.get("starting_stack", 0.0) or 0.0)

    sanitized_players: List[Dict[str, Any]] = [
        {"player_uid": f"seat_{s}", "seat": s,
         "starting_stack": float(seat_to_stack.get(s, 0.0)),
         "hole_cards": None, "showed_hand": False}
        for s in range(1, _DEFAULT_MAX_SEATS + 1)
    ]

    # --- Actions: keep only `street`, zero everything else ---
    raw_actions: List[Dict[str, Any]] = []
    for a in actions_raw:
        if not isinstance(a, dict):
            continue
        raw_actions.append({
            "action_id": "",
            "street": str(a.get("street", "")),
            "actor_seat": 0,
            "action_type": "action",
            "amount": 0.0,
            "raise_to": None,
            "call_to": None,
            "normalized_amount_bb": 0.0,
            "pot_before": 0.0,
            "pot_after": 0.0,
        })

    # --- Normalize to exactly _MINER_ACTION_WINDOW slots ---
    sanitized_actions: List[Dict[str, Any]] = []
    if raw_actions:
        if len(raw_actions) >= _MINER_ACTION_WINDOW:
            last_idx = len(raw_actions) - 1
            if _MINER_ACTION_WINDOW == 1:
                indices = [0]
            else:
                indices = [
                    int(round(i * last_idx / (_MINER_ACTION_WINDOW - 1)))
                    for i in range(_MINER_ACTION_WINDOW)
                ]
            sanitized_actions = [raw_actions[i] for i in indices]
        else:
            sanitized_actions = list(raw_actions)
            last = raw_actions[-1]
            while len(sanitized_actions) < _MINER_ACTION_WINDOW:
                pad = dict(last)
                pad["action_id"] = "pad"
                sanitized_actions.append(pad)

    for idx, a in enumerate(sanitized_actions, start=1):
        a["action_id"] = str(idx)

    if not isinstance(outcome, dict):
        outcome = {}

    return {
        "metadata": {
            "game_type": str(metadata.get("game_type", "")),
            "limit_type": str(metadata.get("limit_type", "")),
            "max_seats": _DEFAULT_MAX_SEATS,
            "hero_seat": 0,
            "hand_ended_on_street": "",
            "button_seat": 0,
            "sb": _SANITIZED_SB,
            "bb": _SANITIZED_BB,
            "ante": _SANITIZED_ANTE,
            "rng_seed_commitment": None,
        },
        "players": sanitized_players,
        "streets": [],
        "actions": sanitized_actions,
        "outcome": {
            "winners": [],
            "payouts": {},
            "total_pot": float(outcome.get("total_pot", 0.0) or 0.0),
            "rake": float(outcome.get("rake", 0.0) or 0.0),
            "result_reason": "",
            "showdown": False,
        },
    }
