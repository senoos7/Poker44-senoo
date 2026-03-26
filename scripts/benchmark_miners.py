"""
Competitive miner benchmark for Poker44.

Simulates a validator sending the SAME chunks to multiple miner strategies
and ranks them by reward score — exactly like the real subnet weight system.

Strategies compared:
  YOUR_ML      — your trained RandomForest model (the real miner)
  heuristic    — variance-based fallback (no model)
  always_bot   — predicts every chunk as BOT (score=1.0)
  always_human — predicts every chunk as HUMAN (score=0.0)
  random       — random scores (dumb baseline)
  threshold_lo — conservative: only flag obvious bots (threshold=0.7)
  threshold_hi — aggressive: flag anything above 0.3

Usage:
    cd ~/Poker44-subnet
    python scripts/benchmark_miners.py
    python scripts/benchmark_miners.py --n-cycles 10 --chunks 40
    python scripts/benchmark_miners.py --seed 999
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from poker44.validator.sanitization import sanitize_hand_for_miner
from poker44.score.scoring import reward
from poker44.miner_model.detector import BotDetector
from hands_generator.data_generator import generate_bot_chunk, _default_bot_profiles

HUMAN_CORPUS = REPO_ROOT / "hands_generator" / "human_hands" / "poker_hands_combined.json.gz"

BOLD  = "\033[1m"
GREEN = "\033[32m"
YELLOW= "\033[33m"
RED   = "\033[31m"
CYAN  = "\033[36m"
DIM   = "\033[2m"
RESET = "\033[0m"


# ─────────────────────────────────────────────────────────────
# Miner strategy definitions
# ─────────────────────────────────────────────────────────────

def _make_strategies(detector: BotDetector, rng: random.Random) -> List[Tuple[str, Callable]]:
    """Return list of (name, scoring_fn) pairs. scoring_fn(chunks) -> List[float]"""

    def your_ml(chunks):
        return [detector.score_chunk(c) for c in chunks]

    def heuristic(chunks):
        d = BotDetector.__new__(BotDetector)
        d._model = None
        d._model_path = Path("/nonexistent")
        return [d._score_heuristic(c) for c in chunks]

    def always_bot(chunks):
        return [1.0] * len(chunks)

    def always_human(chunks):
        return [0.0] * len(chunks)

    def random_scores(chunks):
        return [rng.random() for _ in chunks]

    def threshold_conservative(chunks):
        # Score using ML but only flag very high confidence (reduces FPR risk)
        return [min(s * 1.2, 1.0) if detector.is_model_loaded()
                else detector._score_heuristic(c)
                for s, c in zip(your_ml(chunks), chunks)]

    return [
        ("YOUR_ML",       your_ml),
        ("heuristic",     heuristic),
        ("always_bot",    always_bot),
        ("always_human",  always_human),
        ("random",        random_scores),
        ("conservative",  threshold_conservative),
    ]


# ─────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────

def load_human_hands() -> List[Dict[str, Any]]:
    with gzip.open(HUMAN_CORPUS, "rt") as f:
        return json.load(f)


def build_batch(
    human_hands, n_chunks, hands_per_chunk, human_ratio, rng
) -> Tuple[List[List[Dict]], List[int]]:
    profiles = _default_bot_profiles()
    chunks, labels = [], []
    for _ in range(n_chunks):
        size = rng.randint(max(1, hands_per_chunk - 20), hands_per_chunk + 20)
        if rng.random() < human_ratio:
            raw = rng.sample(human_hands, min(size, len(human_hands)))
            labels.append(0)
        else:
            raw = generate_bot_chunk(size=size, profiles=profiles,
                                     seed=rng.randint(0, 10**9))
            labels.append(1)
        chunks.append([sanitize_hand_for_miner(h) for h in raw])
    return chunks, labels


# ─────────────────────────────────────────────────────────────
# Run benchmark
# ─────────────────────────────────────────────────────────────

def run_benchmark(args: argparse.Namespace) -> None:
    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  Poker44 — Competitive Miner Benchmark{RESET}")
    print(f"{BOLD}{'='*65}{RESET}")
    print(f"  cycles={args.n_cycles}  chunks/cycle={args.chunks}  "
          f"hands≈{args.hands}  human_ratio={args.human_ratio:.0%}  seed={args.seed}")

    detector = BotDetector()
    model_status = "ML ✓" if detector.is_model_loaded() else "heuristic (no model)"
    print(f"  YOUR_ML model: {BOLD}{model_status}{RESET}\n")

    rng_data  = random.Random(args.seed)
    rng_strat = random.Random(args.seed + 1)

    print("  Loading human corpus...")
    human_hands = load_human_hands()
    print(f"  {len(human_hands):,} hands loaded.\n")

    strategies = _make_strategies(detector, rng_strat)
    accum: Dict[str, List[float]] = {name: [] for name, _ in strategies}

    for cycle in range(1, args.n_cycles + 1):
        chunks, labels = build_batch(
            human_hands, args.chunks, args.hands, args.human_ratio, rng_data
        )
        y_true = np.array(labels, dtype=int)
        n_bot = int(y_true.sum())
        n_human = len(labels) - n_bot

        print(f"{CYAN}[Cycle {cycle:>2}/{args.n_cycles}]{RESET} "
              f"chunks={len(chunks)} | bot={n_bot} | human={n_human}")

        for name, fn in strategies:
            scores = fn(chunks)
            y_pred = np.array(scores, dtype=float)
            rew, _ = reward(y_pred, y_true)
            accum[name].append(rew)

        # Per-cycle mini table
        print(f"  {'Strategy':<16} {'reward':>7}")
        for name, _ in strategies:
            r = accum[name][-1]
            col = GREEN if r > 0.5 else (YELLOW if r > 0.1 else RED)
            mine = f"{BOLD} ← YOU{RESET}" if name == "YOUR_ML" else ""
            print(f"  {name:<16} {col}{r:>7.4f}{RESET}{mine}")
        print()

    # ── Final leaderboard ──────────────────────────────────────
    avg_rewards = {name: float(np.mean(v)) for name, v in accum.items()}
    ranked = sorted(avg_rewards.items(), key=lambda x: x[1], reverse=True)

    total_reward = sum(r for _, r in ranked if r > 0) or 1.0

    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  LEADERBOARD  ({args.n_cycles} cycles × {args.chunks} chunks){RESET}")
    print(f"{BOLD}{'='*65}{RESET}")
    print(f"  {'Rank':<5} {'Strategy':<16} {'AvgReward':>10} {'Weight%':>9} {'vs YOU':>9}")
    print(f"  {'─'*55}")

    your_reward = avg_rewards.get("YOUR_ML", 0.0)

    for rank, (name, avg_r) in enumerate(ranked, 1):
        weight_pct = avg_r / total_reward * 100 if avg_r > 0 else 0.0
        vs_you = avg_r - your_reward
        vs_str = f"{vs_you:+.4f}" if name != "YOUR_ML" else "  (you)"

        if name == "YOUR_ML":
            col = CYAN + BOLD
        elif avg_r > your_reward:
            col = GREEN
        elif avg_r == 0:
            col = DIM
        else:
            col = YELLOW

        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank} ")
        print(f"  {medal:<5} {col}{name:<16}{RESET}  "
              f"{col}{avg_r:>10.4f}{RESET}  {weight_pct:>8.2f}%  {vs_str:>9}")

    print(f"  {'─'*55}")

    your_rank = next(i+1 for i, (n, _) in enumerate(ranked) if n == "YOUR_ML")
    your_weight = avg_rewards["YOUR_ML"] / total_reward * 100 if avg_rewards["YOUR_ML"] > 0 else 0

    col = GREEN if your_rank == 1 else (YELLOW if your_rank <= 3 else RED)
    print(f"\n  YOUR_ML rank: {col}{BOLD}#{your_rank} of {len(ranked)}{RESET}  "
          f"| est. weight share: {col}{your_weight:.2f}%{RESET}")
    print(f"{BOLD}{'='*65}{RESET}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Competitive Poker44 miner benchmark")
    p.add_argument("--n-cycles",    type=int,   default=5,   help="Simulation cycles")
    p.add_argument("--chunks",      type=int,   default=40,  help="Chunks per cycle")
    p.add_argument("--hands",       type=int,   default=80,  help="Approx hands per chunk")
    p.add_argument("--human-ratio", type=float, default=0.5, help="Fraction of human chunks")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
