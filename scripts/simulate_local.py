"""
Local validator–miner simulation for Poker44.

Simulates a full validator forward cycle completely offline:
  1. Generates mixed bot/human chunks (same logic as the real validator)
  2. Sanitizes each hand (exactly as sanitize_hand_for_miner does)
  3. Calls the miner's forward() directly — no network, no wallets needed
  4. Scores the response using the real reward function
  5. Prints a detailed performance report

Usage:
    cd /path/to/Poker44-subnet
    python scripts/simulate_local.py
    python scripts/simulate_local.py --n-cycles 5 --chunks 40 --hands 80
    python scripts/simulate_local.py --seed 123
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# --- Poker44 imports ---
from poker44.validator.sanitization import sanitize_hand_for_miner
from poker44.validator.synapse import DetectionSynapse
from poker44.score.scoring import reward
from poker44.miner_model.detector import BotDetector
from hands_generator.data_generator import generate_bot_chunk, _default_bot_profiles

# ─────────────────────────────────────────────────────────────
# Colours for terminal output
# ─────────────────────────────────────────────────────────────
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

HUMAN_CORPUS = REPO_ROOT / "hands_generator" / "human_hands" / "poker_hands_combined.json.gz"


# ─────────────────────────────────────────────────────────────
# Data helpers (mirrors validator's data provider)
# ─────────────────────────────────────────────────────────────

def load_human_hands() -> List[Dict[str, Any]]:
    with gzip.open(HUMAN_CORPUS, "rt", encoding="utf-8") as f:
        return json.load(f)


def build_validator_batch(
    human_hands: List[Dict[str, Any]],
    n_chunks: int,
    hands_per_chunk: int,
    human_ratio: float,
    rng: random.Random,
) -> Tuple[List[List[Dict[str, Any]]], List[int]]:
    """
    Build one validator batch: list of sanitized chunks + ground-truth labels.
    Labels: 1 = bot chunk, 0 = human chunk  (same convention as real validator)
    """
    profiles = _default_bot_profiles()
    chunks: List[List[Dict[str, Any]]] = []
    labels: List[int] = []

    for _ in range(n_chunks):
        is_human = rng.random() < human_ratio
        size = rng.randint(max(1, hands_per_chunk - 20), hands_per_chunk + 20)

        if is_human:
            raw = rng.sample(human_hands, min(size, len(human_hands)))
            label = 0
        else:
            raw = generate_bot_chunk(size=size, profiles=profiles,
                                     seed=rng.randint(0, 10**9))
            label = 1

        # Apply validator sanitization
        sanitized = [sanitize_hand_for_miner(h) for h in raw]
        chunks.append(sanitized)
        labels.append(label)

    return chunks, labels


# ─────────────────────────────────────────────────────────────
# Miner call (in-process — no network)
# ─────────────────────────────────────────────────────────────

async def call_miner_forward(
    detector: BotDetector,
    chunks: List[List[Dict[str, Any]]],
) -> List[float]:
    """Call the miner's scoring logic directly, same as neurons/miner.py forward()."""
    synapse = DetectionSynapse(chunks=chunks)

    scores = [detector.score_chunk(chunk) for chunk in synapse.chunks]
    synapse.risk_scores = scores
    synapse.predictions = [detector.predict_chunk(chunk) for chunk in synapse.chunks]

    return synapse.risk_scores


# ─────────────────────────────────────────────────────────────
# One simulation cycle
# ─────────────────────────────────────────────────────────────

async def run_cycle(
    detector: BotDetector,
    human_hands: List[Dict[str, Any]],
    n_chunks: int,
    hands_per_chunk: int,
    human_ratio: float,
    rng: random.Random,
    cycle_num: int,
) -> dict:
    # 1. Build validator batch
    chunks, labels = build_validator_batch(
        human_hands, n_chunks, hands_per_chunk, human_ratio, rng
    )

    n_bot_chunks   = sum(labels)
    n_human_chunks = len(labels) - n_bot_chunks
    print(f"\n{CYAN}[Cycle {cycle_num}]{RESET} "
          f"chunks={len(chunks)} | bot={n_bot_chunks} | human={n_human_chunks} | "
          f"hands≈{sum(len(c) for c in chunks)}")

    # 2. Call miner forward
    t0 = time.perf_counter()
    scores = await call_miner_forward(detector, chunks)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # 3. Compute reward
    y_pred  = np.array(scores, dtype=float)
    y_true  = np.array(labels, dtype=int)
    rew, metrics = reward(y_pred, y_true)

    # 4. Per-chunk detail
    print(f"\n  {'#':>3}  {'label':>6}  {'score':>7}  {'pred':>6}  {'correct':>8}")
    print(f"  {'-'*40}")
    correct = 0
    for i, (label, score) in enumerate(zip(labels, scores)):
        pred = "BOT" if score >= 0.52 else "HUMAN"
        actual = "BOT" if label == 1 else "HUMAN"
        ok = pred == actual
        if ok:
            correct += 1
        colour = GREEN if ok else RED
        print(f"  {i:>3}  {actual:>6}  {score:>7.3f}  {colour}{pred:>6}{RESET}  "
              f"{'✓' if ok else '✗':>8}")

    # 5. Summary
    accuracy = correct / len(labels)
    colour_rew = GREEN if rew > 0.3 else (YELLOW if rew > 0.05 else RED)
    print(f"\n  {BOLD}── Metrics ──────────────────────────────{RESET}")
    print(f"  accuracy         : {accuracy:.1%}  ({correct}/{len(labels)})")
    print(f"  ap_score         : {metrics['ap_score']:.4f}   (0=worst, 1=best)")
    print(f"  bot_recall       : {metrics['bot_recall']:.4f}   (detected {metrics['bot_recall']:.0%} of bots)")
    print(f"  fpr              : {metrics['fpr']:.4f}   "
          f"{'✓ OK' if metrics['fpr'] < 0.10 else RED + '✗ PENALTY (≥0.10, reward=0)' + RESET}")
    print(f"  human_safety     : {metrics['human_safety_penalty']:.4f}")
    print(f"  {BOLD}reward           : {colour_rew}{rew:.4f}{RESET}")
    print(f"  inference_time   : {elapsed_ms:.1f} ms for {len(chunks)} chunks")

    return {
        "cycle": cycle_num,
        "reward": rew,
        "ap_score": metrics["ap_score"],
        "bot_recall": metrics["bot_recall"],
        "fpr": metrics["fpr"],
        "accuracy": accuracy,
        "elapsed_ms": elapsed_ms,
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Poker44 — Local Validator/Miner Simulation{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"  cycles         : {args.n_cycles}")
    print(f"  chunks/cycle   : {args.chunks}")
    print(f"  hands/chunk    : ~{args.hands}")
    print(f"  human ratio    : {args.human_ratio:.0%}")
    print(f"  seed           : {args.seed}")

    # Load detector
    detector = BotDetector()
    model_type = "ML" if detector.is_model_loaded() else "heuristic"
    print(f"  model          : {BOLD}{model_type}{RESET}")
    if not detector.is_model_loaded():
        print(f"  {YELLOW}⚠ No model.pkl found — using heuristic. "
              f"Run: python -m poker44.miner_model.train{RESET}")

    # Load human corpus
    print(f"\n  Loading human corpus from {HUMAN_CORPUS.name}...")
    human_hands = load_human_hands()
    print(f"  {len(human_hands):,} human hands loaded.")

    rng = random.Random(args.seed)
    results = []

    for i in range(1, args.n_cycles + 1):
        result = await run_cycle(
            detector=detector,
            human_hands=human_hands,
            n_chunks=args.chunks,
            hands_per_chunk=args.hands,
            human_ratio=args.human_ratio,
            rng=rng,
            cycle_num=i,
        )
        results.append(result)

    # ── Overall summary ──────────────────────────────────────────
    if args.n_cycles > 1:
        avg_reward   = np.mean([r["reward"] for r in results])
        avg_ap       = np.mean([r["ap_score"] for r in results])
        avg_recall   = np.mean([r["bot_recall"] for r in results])
        avg_fpr      = np.mean([r["fpr"] for r in results])
        avg_accuracy = np.mean([r["accuracy"] for r in results])
        avg_ms       = np.mean([r["elapsed_ms"] for r in results])

        colour_avg = GREEN if avg_reward > 0.3 else (YELLOW if avg_reward > 0.05 else RED)
        print(f"\n{BOLD}{'='*60}{RESET}")
        print(f"{BOLD}  Overall ({args.n_cycles} cycles){RESET}")
        print(f"{'='*60}")
        print(f"  avg reward     : {colour_avg}{avg_reward:.4f}{RESET}")
        print(f"  avg AP         : {avg_ap:.4f}")
        print(f"  avg bot_recall : {avg_recall:.4f}")
        print(f"  avg FPR        : {avg_fpr:.4f}  "
              f"({'✓ safe' if avg_fpr < 0.10 else RED + '✗ risky' + RESET})")
        print(f"  avg accuracy   : {avg_accuracy:.1%}")
        print(f"  avg latency    : {avg_ms:.1f} ms/cycle")
        print(f"{'='*60}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local Poker44 validator/miner simulation")
    p.add_argument("--n-cycles",     type=int,   default=3,    help="Number of forward cycles to simulate")
    p.add_argument("--chunks",       type=int,   default=40,   help="Chunks per cycle (real validator uses 40-60)")
    p.add_argument("--hands",        type=int,   default=80,   help="Approx hands per chunk (real: 60-100)")
    p.add_argument("--human-ratio",  type=float, default=0.5,  help="Fraction of human chunks (real: 0.4-0.6)")
    p.add_argument("--seed",         type=int,   default=42,   help="Random seed for reproducibility")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
