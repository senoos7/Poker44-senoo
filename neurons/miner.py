"""Poker44 miner — ML bot detector blended with reference behavioral heuristic."""

import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

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

        # Public manifest version MUST match the actual loaded model folder
        # so that the published identity is verifiable in the repo at
        # `repo_commit` (per the model integrity policy). The run script sets
        # POKER44_MODEL_VERSION = MODEL_VERSION; we fall back to the same
        # MODEL_VERSION value (and ultimately the detector's label) so direct
        # `python neurons/miner.py` invocations stay compliant too.
        _model_version = (
            os.getenv("POKER44_MODEL_VERSION")
            or os.getenv("MODEL_VERSION")
            or self._detector.model_label
            or "default"
        ).strip() or "default"

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
                "model_name": "poker44-ml-heuristic",
                "model_version": _model_version,
                "framework": "scikit-learn+heuristic",
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
                "notes": "HistGBM bot-detector trained on Poker44 public benchmark data.",
                # open_source controls the integrity check:
                #   True  → validator fetches repo_url + repo_commit → PUBLIC repo required
                #   False → miner shows as "opaque" (currently NOT zeroed per owner policy)
                # Set POKER44_MODEL_OPEN_SOURCE=true (or flip default) only after making
                # https://github.com/senoos7/Poker44-senoo public.
                "open_source": os.getenv("POKER44_MODEL_OPEN_SOURCE", "false").strip().lower()
                               in {"1", "true", "yes"},
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
        status = self.manifest_compliance["status"]
        missing = self.manifest_compliance["missing_fields"]
        os_flag = self.model_manifest.get("open_source")
        # evaluate_manifest_compliance puts 'open_source' in missing_fields when
        # open_source is False — meaning "not transparent", not "field absent".
        if status == "opaque" and missing == ["open_source"] and os_flag is False:
            bt.logging.info(
                "Miner transparency: opaque (open_source=false). "
                "Validators skip public-repo integrity checks; expected while repo is private."
            )
        else:
            bt.logging.info(
                f"Miner transparency status: {status} "
                f"(missing_fields={missing}, policy_violations="
                f"{self.manifest_compliance.get('policy_violations', [])})"
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
            "Miner prep docs available | "
            f"miner_doc={repo_root / 'docs' / 'miner.md'}"
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

        import time
        import traceback

        try:
            # Batch-score all chunks in a single predict_proba call.
            # This is much faster than run_in_executor with 40 individual calls
            # and avoids event-loop contention with background metagraph sync.
            t0 = time.monotonic()
            ml_scores = self._detector.score_chunks_batch(chunks)
            heuristic_scores = [self._score_chunk_heuristic(chunk) for chunk in chunks]
            blended = [self._blend(ml, h) for ml, h in zip(ml_scores, heuristic_scores)]

            # Within-batch rank stretch: fixes "all chunks ≈ 0.35 → all HUMAN"
            # collapse on shifted live data. The model's ordering is usually
            # correct even when its absolute confidence is compressed; this
            # remaps the blended scores to use the full [0, 1] range while
            # keeping the original ordering, so AP / AUROC see a proper
            # discrimination signal and the BOT/HUMAN split tracks the actual
            # batch composition (typically near 50/50).
            scores = self._batch_rank_stretch(blended)

            elapsed_ms = int((time.monotonic() - t0) * 1000)
            synapse.risk_scores = scores
            predictions = [score >= 0.5 for score in scores]
            synapse.predictions = predictions
            synapse.model_manifest = dict(self.model_manifest)
            bt.logging.debug(f"[INFERENCE] {elapsed_ms}ms for {len(chunks)} chunks")
        except Exception as exc:
            bt.logging.error(
                f"[FORWARD ERROR] scoring failed for {len(chunks)} chunks: {exc}\n"
                f"{traceback.format_exc()}"
            )
            synapse.risk_scores = [0.5] * len(chunks)
            synapse.predictions = [False] * len(chunks)
            return synapse

        n_bot = sum(predictions)
        n_human = len(predictions) - n_bot

        # --- INFO: response summary ---
        bt.logging.info(
            f"[RESPONSE] chunks={len(chunks)} | predicted_bot={n_bot} | predicted_human={n_human} | "
            f"ml_scores={[round(s, 3) for s in ml_scores]} | "
            f"blended={[round(s, 3) for s in blended]} | "
            f"final={[round(s, 3) for s in scores]}"
        )

        # --- DEBUG: full per-chunk detail ---
        for i, (chunk, score, pred) in enumerate(zip(chunks, scores, predictions)):
            bt.logging.debug(
                f"[CHUNK {i:02d}] hands={len(chunk)} | score={score:.4f} | prediction={'BOT' if pred else 'HUMAN'}"
            )

        return synapse

    # ------------------------------------------------------------------
    # Reference behavioral heuristic (from Poker44 v1 baseline)
    # Gets ~0.53 composite on its own. Used to floor/blend with ML scores
    # so we always produce a useful ranking even when ML is uncertain.
    # ------------------------------------------------------------------
    @staticmethod
    def _clamp01(v: float) -> float:
        return max(0.0, min(1.0, v))

    @classmethod
    def _batch_rank_stretch(
        cls,
        scores: List[float],
        *,
        rank_weight: float = 0.65,
        std_floor: float = 0.04,
    ) -> List[float]:
        """Stretch a batch of scores to use the full [0, 1] range.

        - Computes percentile rank within the batch (ties get the same rank).
        - Final score = rank_weight * rank + (1 - rank_weight) * raw_score.
        - When the batch already has high variance (std > std_floor) we keep
          most of the raw score; when scores are bunched (current failure
          mode), we lean on rank position to surface the model's ordering.

        The relative ordering produced by the ML+heuristic blend is preserved,
        so AP / AUROC are at worst unchanged and typically improve.
        """
        n = len(scores)
        if n <= 1:
            return [round(cls._clamp01(s), 6) for s in scores]

        mean_s = sum(scores) / n
        var_s = sum((s - mean_s) ** 2 for s in scores) / n
        std_s = var_s ** 0.5

        # Adaptive blend: when scores already span a wide range, trust them
        # more; when they're bunched, lean on rank.
        if std_s >= std_floor * 4:
            effective_rank_w = rank_weight * 0.3
        elif std_s >= std_floor * 2:
            effective_rank_w = rank_weight * 0.6
        else:
            effective_rank_w = rank_weight

        # Tie-aware percentile rank (average rank for ties)
        sorted_idx = sorted(range(n), key=lambda i: scores[i])
        rank_pos = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and scores[sorted_idx[j + 1]] == scores[sorted_idx[i]]:
                j += 1
            avg_rank = (i + j) / 2.0
            for k in range(i, j + 1):
                rank_pos[sorted_idx[k]] = avg_rank / max(n - 1, 1)
            i = j + 1

        out = [
            cls._clamp01(
                (1.0 - effective_rank_w) * scores[i] + effective_rank_w * rank_pos[i]
            )
            for i in range(n)
        ]
        return [round(v, 6) for v in out]

    @classmethod
    def _score_hand_heuristic(cls, hand: dict) -> float:
        actions = hand.get("actions") or []
        players = hand.get("players") or []
        streets = hand.get("streets") or []
        outcome = hand.get("outcome") or {}

        action_counts = Counter(a.get("action_type") for a in actions)
        meaningful = max(1, sum(
            action_counts.get(k, 0)
            for k in ("call", "check", "bet", "raise", "fold")
        ))

        call_ratio  = action_counts.get("call",  0) / meaningful
        check_ratio = action_counts.get("check", 0) / meaningful
        fold_ratio  = action_counts.get("fold",  0) / meaningful
        raise_ratio = action_counts.get("raise", 0) / meaningful
        street_depth    = len(streets) / 3.0
        showdown_flag   = 1.0 if outcome.get("showdown") else 0.0
        player_signal   = (6 - min(len(players), 6)) / 4.0 if players else 0.0

        s = (0.32 * street_depth
             + 0.22 * showdown_flag
             + 0.18 * cls._clamp01(call_ratio  / 0.35)
             + 0.12 * cls._clamp01(check_ratio / 0.30)
             + 0.08 * player_signal
             - 0.18 * cls._clamp01(fold_ratio  / 0.55)
             - 0.10 * cls._clamp01(raise_ratio / 0.20))
        return cls._clamp01(s)

    @classmethod
    def _score_chunk_heuristic(cls, chunk: List[dict]) -> float:
        """Chunk-level heuristic bot score in [0, 1].

        Captures two primary signals that remain discriminative even for
        humanized bots:

        1. Bet-size mechanical repetition (LOW coefficient of variation of
           non-zero bet amounts across ALL hands in the chunk). Humanized bots
           still tend to reuse the same bet amounts (e.g. always 2BB, 3BB).

        2. Within-chunk behavioral consistency (LOW std of per-hand heuristic
           scores). Even humanized bots are mechanically consistent across
           hands; genuine human sessions have high variance.
        """
        if not chunk:
            return 0.5

        # Per-hand base scores (fold-heavy / shallow → lower scores)
        hand_scores = [cls._score_hand_heuristic(h) for h in chunk]
        mean_score = sum(hand_scores) / len(hand_scores)

        # --- Signal 1: within-chunk behavioral consistency ---
        # LOW std → mechanical consistency → bot signal (HIGH score)
        if len(hand_scores) > 2:
            std_scores = (
                sum((s - mean_score) ** 2 for s in hand_scores) / len(hand_scores)
            ) ** 0.5
            # std typically 0.05–0.08 for bots, 0.15–0.25 for humans
            # map std=0 → 1.0 (certain bot), std≥0.20 → 0.0 (certain human)
            consistency_signal = max(0.0, 1.0 - std_scores / 0.20)
        else:
            consistency_signal = 0.5

        # --- Signal 2: bet-size mechanical repetition ---
        all_amounts: list = []
        for hand in chunk:
            for action in (hand.get("actions") or []):
                v = action.get("normalized_amount_bb")
                try:
                    fv = float(v) if v is not None else 0.0
                except (TypeError, ValueError):
                    fv = 0.0
                if fv > 0.0:
                    all_amounts.append(fv)

        if len(all_amounts) >= 4:
            mean_amt = sum(all_amounts) / len(all_amounts)
            std_amt = (
                sum((v - mean_amt) ** 2 for v in all_amounts) / len(all_amounts)
            ) ** 0.5
            bet_cv = std_amt / max(mean_amt, 1e-6)
            # LOW bet_cv (< 0.3) → mechanical → bot; HIGH bet_cv (> 1.5) → human
            # map bet_cv=0 → 1.0 (certain bot), bet_cv≥1.5 → 0.0 (certain human)
            bet_consistency = max(0.0, 1.0 - min(bet_cv, 1.5) / 1.5)
        else:
            bet_consistency = 0.5

        # Blend: consistency signals dominate; mean per-hand score provides a
        # weak prior aligned with old-style bot patterns.
        score = (
            0.45 * consistency_signal
            + 0.45 * bet_consistency
            + 0.10 * mean_score
        )
        return round(cls._clamp01(score), 6)

    @classmethod
    def _blend(cls, ml: float, heuristic: float, ml_weight: float = 0.6) -> float:
        """Weighted blend of ML and heuristic scores.

        ml_weight=0.6 means ML drives 60% of the final score; the heuristic
        provides a 40% floor so we always produce meaningful rankings even
        when the ML model is uncertain on unseen real-world data.
        """
        return round(cls._clamp01(ml_weight * ml + (1.0 - ml_weight) * heuristic), 6)

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
