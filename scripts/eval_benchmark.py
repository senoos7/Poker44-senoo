"""
eval_benchmark.py — evaluate a miner model against the public benchmark.

Loads data/public_miner_benchmark.json.gz (or a custom path) and scores
every chunk with the BotDetector, then reports the same metrics the
validator uses for rewards:

  reward = (0.65 * AP + 0.35 * bot_recall) * max(0, 1 - FPR)^2
           → 0 if FPR >= 0.10

Usage:
  # Evaluate whatever MODEL_VERSION env var says (default: v1_rf_synthetic)
  cd /path/to/Poker44-subnet
  python scripts/eval_benchmark.py

  # Evaluate a specific version
  MODEL_VERSION=v2_rf_mixed python scripts/eval_benchmark.py

  # Compare two versions side-by-side
  python scripts/eval_benchmark.py --compare v1_rf_synthetic v2_rf_mixed

  # Use a custom benchmark file
  python scripts/eval_benchmark.py --benchmark /tmp/custom_benchmark.json.gz

  # Only score the validation split (default: all splits)
  python scripts/eval_benchmark.py --split validation
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BENCHMARK = REPO_ROOT / "data" / "public_miner_benchmark.json.gz"


# ------------------------------------------------------------------
# Benchmark loader
# ------------------------------------------------------------------

def load_benchmark(path: Path) -> List[Dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    chunks = payload.get("labeled_chunks", [])
    print(f"  Loaded benchmark: {len(chunks)} chunks  (hash={payload.get('dataset_hash','?')[:12]}...)")
    stats = payload.get("stats", {})
    print(f"  Bot chunks: {stats.get('bot_chunks','?')}  "
          f"Human chunks: {stats.get('human_chunks','?')}  "
          f"Total hands: {stats.get('total_hands','?')}  "
          f"Shortcut-rule accuracy: {stats.get('shortcut_rule_accuracy','?'):.3f}")
    return chunks


# ------------------------------------------------------------------
# Metrics (mirrors poker44/score/scoring.py reward function)
# ------------------------------------------------------------------

def compute_metrics(
    chunks: List[Dict[str, Any]],
    scores: List[float],
    threshold: float = 0.52,
) -> Dict[str, float]:
    from sklearn.metrics import average_precision_score

    labels   = np.array([1 if c["is_bot"] else 0 for c in chunks], dtype=int)
    scores_a = np.array(scores, dtype=float)
    preds    = (scores_a >= threshold).astype(int)

    n_bot   = int(labels.sum())
    n_human = int((1 - labels).sum())

    if n_bot == 0 or n_human == 0:
        return {"error": "benchmark has no bot or no human chunks"}

    ap = float(average_precision_score(labels, scores_a))

    # Bot recall = TP / (TP + FN)  [how many bots we caught]
    tp = int(((preds == 1) & (labels == 1)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    bot_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # FPR = FP / (FP + TN)  [how often we falsely flag humans as bots]
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Reward formula
    human_safety_penalty = max(0.0, 1.0 - fpr) ** 2
    if fpr >= 0.10:
        reward = 0.0
    else:
        reward = (0.65 * ap + 0.35 * bot_recall) * human_safety_penalty

    accuracy = float(((preds == labels)).sum()) / len(labels)

    return {
        "ap":           round(ap, 4),
        "bot_recall":   round(bot_recall, 4),
        "fpr":          round(fpr, 4),
        "accuracy":     round(accuracy, 4),
        "reward":       round(reward, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_bot": n_bot, "n_human": n_human,
        "n_chunks": len(chunks),
    }


# ------------------------------------------------------------------
# Evaluate one model version
# ------------------------------------------------------------------

def evaluate_version(
    version: str,
    chunks: List[Dict[str, Any]],
    split: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    if split:
        chunks = [c for c in chunks if c.get("split") == split]
        if not chunks:
            print(f"  No chunks found for split={split!r}")
            return None

    # Set MODEL_VERSION env var so detector loads the right model
    original_env = os.environ.get("MODEL_VERSION")
    os.environ["MODEL_VERSION"] = version

    # Re-import detector with the new env var
    # (detector reads env at instantiation time)
    from poker44.miner_model.detector import BotDetector
    detector = BotDetector()

    if not detector.is_model_loaded():
        print(f"  [{version}] No model.pkl found — skipping")
        if original_env is None:
            os.environ.pop("MODEL_VERSION", None)
        else:
            os.environ["MODEL_VERSION"] = original_env
        return None

    scores = [detector.score_chunk(c["hands"]) for c in chunks]

    if original_env is None:
        os.environ.pop("MODEL_VERSION", None)
    else:
        os.environ["MODEL_VERSION"] = original_env

    return compute_metrics(chunks, scores)


# ------------------------------------------------------------------
# Pretty print
# ------------------------------------------------------------------

def _bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def print_result(version: str, m: Dict[str, float], split_label: str) -> None:
    reward = m["reward"]
    color_reward = (
        "🟢" if reward >= 0.60 else
        "🟡" if reward >= 0.40 else
        "🔴"
    )
    fpr_flag = " ⚠️ HIGH FPR" if m["fpr"] >= 0.10 else ""

    print(f"\n  ── {version}  [{split_label}] ──────────────────────────────")
    print(f"  Reward  {color_reward}  {reward:.4f}   {_bar(min(reward, 1.0))}{fpr_flag}")
    print(f"  AP          {m['ap']:.4f}   {_bar(m['ap'])}")
    print(f"  Bot Recall  {m['bot_recall']:.4f}   {_bar(m['bot_recall'])}")
    print(f"  FPR         {m['fpr']:.4f}   {_bar(m['fpr'])}")
    print(f"  Accuracy    {m['accuracy']:.4f}   {_bar(m['accuracy'])}")
    print(f"  Chunks: {m['n_chunks']} total  "
          f"(bot={m['n_bot']} human={m['n_human']})  "
          f"TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")


def print_comparison(results: Dict[str, Dict]) -> None:
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  {'Version':<25} {'Reward':>7}  {'AP':>7}  {'Recall':>7}  {'FPR':>7}")
    print(f"  {'-'*25} {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

    sorted_versions = sorted(results.items(), key=lambda x: x[1].get("reward", 0), reverse=True)
    for i, (ver, m) in enumerate(sorted_versions):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
        fpr_warn = " ⚠️" if m.get("fpr", 0) >= 0.10 else ""
        print(f"  {medal} {ver:<23} {m.get('reward',0):>7.4f}  "
              f"{m.get('ap',0):>7.4f}  {m.get('bot_recall',0):>7.4f}  "
              f"{m.get('fpr',0):>7.4f}{fpr_warn}")
    print("=" * 60)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate miner model(s) against public benchmark")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK,
                        help="Path to public benchmark JSON.GZ")
    parser.add_argument("--split", choices=["train", "validation", "all"], default="all",
                        help="Which split to evaluate on (default: all)")
    parser.add_argument("--compare", nargs="+", metavar="VERSION",
                        help="Compare multiple versions side-by-side, e.g. --compare v1_rf_synthetic v2_rf_mixed")
    parser.add_argument("--version", type=str, default=None,
                        help="Model version to evaluate (overrides MODEL_VERSION env var)")
    args = parser.parse_args()

    if not args.benchmark.exists():
        print(f"Benchmark not found: {args.benchmark}")
        print("Build it first with:")
        print("  python scripts/publish/publish_public_benchmark.py --skip-wandb")
        sys.exit(1)

    split_label = args.split if args.split != "all" else "train+validation"
    split_filter = None if args.split == "all" else args.split

    print("=" * 60)
    print("Poker44 — Public Benchmark Evaluation")
    print(f"  benchmark : {args.benchmark}")
    print(f"  split     : {split_label}")
    print("=" * 60)

    chunks = load_benchmark(args.benchmark)

    versions_to_test = args.compare or [
        args.version or os.environ.get("MODEL_VERSION", "v1_rf_synthetic")
    ]

    all_results: Dict[str, Dict] = {}
    for ver in versions_to_test:
        print(f"\nEvaluating [{ver}]...")
        m = evaluate_version(ver, chunks, split=split_filter)
        if m:
            print_result(ver, m, split_label)
            all_results[ver] = m

    if len(all_results) >= 2:
        print_comparison(all_results)
    elif len(all_results) == 0:
        print("\nNo models evaluated. Make sure model.pkl files exist.")
        print("Run: python -m poker44.miner_model.train --version <version>")


if __name__ == "__main__":
    main()
