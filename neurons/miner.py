"""Poker44 miner — ML-based bot detector with heuristic fallback."""

import json
import time
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.miner_model.detector import BotDetector
from poker44.validator.synapse import DetectionSynapse


class Miner(BaseMinerNeuron):
    """
    Competitive Poker44 miner.

    Uses a trained GradientBoostingClassifier on chunk-level statistical
    features extracted from the sanitized hand data received from validators.

    Falls back to a calibrated variance-based heuristic if no model file
    exists yet (train one with: python -m poker44.miner_model.train).

    Key insight: bot chunks have LOW within-chunk variance (systematic
    profiles), while human chunks are diverse → high variance.
    """

    def __init__(self, config=None):
        super().__init__(config=config)
        self._detector = BotDetector()
        if self._detector.is_model_loaded():
            bt.logging.info("BotDetector: ML model loaded.")
        else:
            bt.logging.warning(
                "BotDetector: no model found — using heuristic fallback. "
                "Run `python -m poker44.miner_model.train` to train a model."
            )
        bt.logging.info(f"Axon created: {self.axon}")

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []

        validator_hotkey = (synapse.dendrite.hotkey if synapse.dendrite else "unknown")
        chunk_sizes = [len(c) for c in chunks]
        total_hands = sum(chunk_sizes)

        # --- INFO: always visible summary ---
        bt.logging.info(
            f"[QUERY] from={validator_hotkey} | chunks={len(chunks)} | hands={total_hands} | "
            f"model={'ML' if self._detector.is_model_loaded() else 'heuristic'}"
        )

        # --- DEBUG: per-chunk sizes ---
        bt.logging.debug(
            f"[QUERY CHUNKS] sizes={chunk_sizes}"
        )

        # --- DEBUG: sample hand from the first chunk (shows available fields) ---
        if chunks and chunks[0]:
            sample_hand = chunks[0][0]
            sample_keys = list(sample_hand.keys())
            bt.logging.debug(
                f"[QUERY SAMPLE HAND] keys={sample_keys}"
            )
            bt.logging.debug(
                f"[QUERY SAMPLE HAND] data={json.dumps(sample_hand, default=str)}"
            )

        scores = [self._detector.score_chunk(chunk) for chunk in chunks]
        synapse.risk_scores = scores
        predictions = [self._detector.predict_chunk(chunk) for chunk in chunks]
        synapse.predictions = predictions

        n_bot = sum(predictions)
        n_human = len(predictions) - n_bot

        # --- INFO: response summary ---
        bt.logging.info(
            f"[RESPONSE] chunks={len(chunks)} | predicted_bot={n_bot} | predicted_human={n_human} | "
            f"scores={[round(s, 3) for s in scores]}"
        )

        # --- DEBUG: full per-chunk detail ---
        for i, (chunk, score, pred) in enumerate(zip(chunks, scores, predictions)):
            bt.logging.debug(
                f"[CHUNK {i:02d}] hands={len(chunk)} | score={score:.4f} | prediction={'BOT' if pred else 'HUMAN'}"
            )

        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        incoming = (synapse.dendrite.hotkey if synapse.dendrite else "unknown")
        bt.logging.info(f"[BLACKLIST CHECK] incoming hotkey={incoming}")
        blocked, reason = self.common_blacklist(synapse)
        if blocked:
            bt.logging.warning(f"[BLACKLIST REJECTED] hotkey={incoming} reason={reason}")
        else:
            bt.logging.info(f"[BLACKLIST ALLOWED] hotkey={incoming} reason={reason}")
        return blocked, reason

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(
                f"UID={miner.uid} | "
                f"incentive={miner.metagraph.I[miner.uid]:.6f} | "
                f"model={'ML' if miner._detector.is_model_loaded() else 'heuristic'}"
            )
            time.sleep(5 * 60)
