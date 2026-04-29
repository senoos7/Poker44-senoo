"""Poker44 miner — ML-based bot detector with heuristic fallback."""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.miner_model.detector import BotDetector
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
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
            bt.logging.info(f"BotDetector: ML model loaded ({self._detector.model_label}).")
        else:
            bt.logging.warning(
                "BotDetector: no model found — using heuristic fallback. "
                "Run `python -m poker44.miner_model.train --version <version>` to train a model."
            )
        repo_root = Path(__file__).resolve().parents[1]

        # Auto-detect the current git commit so the manifest carries a real
        # repo_commit value.  Without this the validator marks every miner
        # as opaque and logs manifest_missing_repo_commit suspicion on every
        # forward cycle.  The env var POKER44_MODEL_REPO_COMMIT overrides.
        try:
            _auto_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_root),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            _auto_commit = ""

        # Use the MODEL_VERSION env var (set by run scripts) so the manifest
        # version field matches the actual model file being loaded.
        _model_version = os.getenv("MODEL_VERSION", self._detector.model_label or "v4_rf_mixed")

        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            # Include all files that define the miner's actual behaviour.
            # The implementation_sha256 is computed over all of these, making
            # the manifest auditable: anyone can verify the exact code that
            # produced any given response.
            implementation_files=[
                Path(__file__).resolve(),                                      # neurons/miner.py
                repo_root / "poker44" / "miner_model" / "detector.py",        # model loader + heuristic
                repo_root / "poker44" / "miner_model" / "features.py",        # feature engineering
                repo_root / "poker44" / "miner_model" / "train.py",           # training pipeline
                repo_root / "poker44" / "miner_model" / "sanitize.py",        # sanitization wrapper
            ],
            defaults={
                "model_name": "poker44-rf-bot-detector",
                "model_version": _model_version,
                "framework": "scikit-learn",
                "license": "MIT",
                # ⚠️  Must point to YOUR OWN fork/repo, NOT the reference subnet repo.
                # Using the reference repo URL (Poker44/Poker44-subnet) with a custom
                # model_name triggers the repo_url_must_point_to_model_repo policy
                # violation and marks the miner as opaque.
                "repo_url": os.getenv(
                    "POKER44_MODEL_REPO_URL",
                    "https://github.com/senoos7/Poker44-senoo",
                ),
                "repo_commit": _auto_commit,
                "notes": "RandomForest bot-detector trained on mixed human/bot chunk data.",
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained on synthetic and mixed human/bot chunk data derived from the "
                    "Poker44 public benchmark dataset using poker44.miner_model.train."
                ),
                "training_data_sources": ["poker44-public-benchmark"],
                "private_data_attestation": (
                    "This reference miner does not train on validator-only evaluation data."
                ),
            },
        )
        bt.logging.info(f"Published model manifest: {self.model_manifest}")
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)
        
        # # Attach handlers after initialization
        # self.axon.attach(
        #     forward_fn = self.forward,
        #     blacklist_fn = self.blacklist,
        #     priority_fn = self.priority,
        # )
        # bt.logging.info("Attaching forward function to miner axon.")
        
        bt.logging.info(f"Axon created: {self.axon}")

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        bt.logging.info(
            "Miner prep tooling available | "
            f"benchmark_doc={repo_root / 'docs' / 'public-benchmark.md'} "
            f"miner_doc={repo_root / 'docs' / 'miner.md'} "
            f"anti_leakage_doc={repo_root / 'docs' / 'anti-leakage.md'}"
        )
        bt.logging.info(
            "Public benchmark command: "
            "python scripts/publish/publish_public_benchmark.py --skip-wandb"
        )
        bt.logging.info(
            "Purpose: train, validate and refine miner models against the public benchmark "
            "while Poker44 moves toward more dynamic evaluation."
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []

        validator_hotkey = (synapse.dendrite.hotkey if synapse.dendrite else "unknown")
        chunk_sizes = [len(c) for c in chunks]
        total_hands = sum(chunk_sizes)

        # --- INFO: always visible summary ---
        bt.logging.info(
            f"[QUERY] from={validator_hotkey} | chunks={len(chunks)} | hands={total_hands} | "
            f"model={self._detector.model_label}"
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

        import asyncio
        import traceback
        loop = asyncio.get_event_loop()

        try:
            # Run CPU-bound inference in a thread pool so the asyncio event loop
            # stays responsive (handles cancellation, heartbeats, other tasks).
            # Without this, HistGBM inference blocks the entire event loop and the
            # validator's timeout fires before the response is sent back.
            scores = await loop.run_in_executor(
                None,
                lambda: [self._detector.score_chunk(chunk) for chunk in chunks],
            )
            synapse.risk_scores = scores
            predictions = [score >= 0.5 for score in scores]
            synapse.predictions = predictions
            synapse.model_manifest = dict(self.model_manifest)
        except asyncio.CancelledError:
            bt.logging.warning(
                f"[FORWARD CANCELLED] validator timed out after sending {len(chunks)} chunks "
                f"— consider reducing VPS load or validator timeout"
            )
            raise
        except Exception as exc:
            bt.logging.error(
                f"[FORWARD ERROR] scoring failed for {len(chunks)} chunks: {exc}\n"
                f"{traceback.format_exc()}"
            )
            synapse.risk_scores = [0.0] * len(chunks)
            synapse.predictions = [False] * len(chunks)
            return synapse

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
                f"model={miner._detector.model_label}"
            )
            time.sleep(5 * 60)
