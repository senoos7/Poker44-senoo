#!/usr/bin/env python3
"""
Download Poker44 public training benchmark data.

Fetches all released benchmark batches from the official API and saves them
as a single compressed JSON file ready for use in train.py.

Usage:
    python scripts/download_benchmark.py
    python scripts/download_benchmark.py --out data/benchmark.json.gz
    python scripts/download_benchmark.py --out data/benchmark.json.gz --force

Output format (saved to --out):
    {
        "meta": {
            "downloaded_at": "...",
            "release_version": "...",
            "total_chunks": N,
            "total_hands": N,
            "dates": ["2026-05-01", ...]
        },
        "chunks":      [[hand, ...], [hand, ...], ...],  # one item = one scoring unit
        "groundTruth": [0, 1, 0, 1, ...]                # 0=human, 1=bot
    }
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

API_BASE = "https://api.poker44.net/api/v1/benchmark"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "benchmark.json.gz"


def _get(path: str, retries: int = 3) -> Any:
    url = f"{API_BASE}{path}"
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"GET {url} failed after {retries} attempts: {exc}") from exc
            print(f"  Retry {attempt}/{retries} for {url}: {exc}")
            time.sleep(3 * attempt)


def fetch_releases() -> List[Dict]:
    resp = _get("/releases")
    return resp["data"]["releases"]


def fetch_date_chunks(source_date: str) -> Tuple[List[List], List[int]]:
    """Return (all_chunks, all_labels) for one sourceDate.

    The API may return multiple batches per day (pagination via nextCursor).
    Each batch contains ~40 chunks with matching groundTruth labels.
    """
    all_chunks: List[List] = []
    all_labels: List[int] = []
    cursor = None
    page = 0

    while True:
        page += 1
        path = f"/chunks?sourceDate={source_date}"
        if cursor:
            path += f"&cursor={cursor}"

        resp = _get(path)
        batches = resp["data"]["chunks"]

        for batch in batches:
            inner_chunks = batch["chunks"]
            ground_truth = batch["groundTruth"]
            if len(inner_chunks) != len(ground_truth):
                print(f"  WARNING: chunk/label count mismatch in batch {batch.get('chunkId')}: "
                      f"{len(inner_chunks)} vs {len(ground_truth)}")
                min_len = min(len(inner_chunks), len(ground_truth))
                inner_chunks = inner_chunks[:min_len]
                ground_truth = ground_truth[:min_len]
            all_chunks.extend(inner_chunks)
            all_labels.extend(ground_truth)

        cursor = resp["data"].get("nextCursor")
        if not cursor:
            break

    return all_chunks, all_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Poker44 benchmark data")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help=f"Output path (default: {DEFAULT_OUT})")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if output file exists")
    parser.add_argument("--dates", nargs="*",
                        help="Only download specific dates (default: all available)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.force:
        print(f"Output already exists: {out_path}")
        print("Use --force to re-download.")
        # Still print summary
        with gzip.open(out_path, "rt") as f:
            existing = json.load(f)
        m = existing["meta"]
        print(f"  chunks={len(existing['chunks'])}  hands={m['total_hands']}"
              f"  dates={m['dates']}")
        return

    print("=" * 60)
    print("  Poker44 Benchmark Downloader")
    print("=" * 60)

    # Step 1: get available releases
    print("\n[1/3] Fetching release index...")
    releases = fetch_releases()
    print(f"  Available dates: {[r['sourceDate'] for r in releases]}")

    if args.dates:
        releases = [r for r in releases if r["sourceDate"] in args.dates]
        print(f"  Filtered to: {[r['sourceDate'] for r in releases]}")

    if not releases:
        print("No matching releases found.")
        sys.exit(1)

    # Step 2: download each date
    all_chunks: List[List] = []
    all_labels: List[int] = []
    fetched_dates: List[str] = []
    total_hands = 0

    print(f"\n[2/3] Downloading {len(releases)} date(s)...")
    for rel in releases:
        date = rel["sourceDate"]
        expected_chunks = rel["chunkCount"]
        expected_hands = rel["handCount"]
        print(f"  {date}: {expected_chunks} chunks, {expected_hands:,} hands ...", end="", flush=True)

        chunks, labels = fetch_date_chunks(date)
        bot_count = sum(labels)
        human_count = len(labels) - bot_count
        hand_count = sum(len(c) for c in chunks)
        print(f" got {len(chunks)} chunks ({bot_count} bot / {human_count} human), {hand_count:,} hands")

        all_chunks.extend(chunks)
        all_labels.extend(labels)
        fetched_dates.append(date)
        total_hands += hand_count

    # Step 3: save
    print(f"\n[3/3] Saving to {out_path} ...")
    total_bot = sum(all_labels)
    total_human = len(all_labels) - total_bot
    print(f"  Total: {len(all_chunks)} chunks ({total_bot} bot / {total_human} human), {total_hands:,} hands")

    output = {
        "meta": {
            "downloaded_at": datetime.datetime.utcnow().isoformat() + "Z",
            "release_version": releases[0].get("releaseVersion", "v1.1"),
            "total_chunks": len(all_chunks),
            "total_hands": total_hands,
            "n_bot": total_bot,
            "n_human": total_human,
            "dates": sorted(fetched_dates),
        },
        "chunks": all_chunks,
        "groundTruth": all_labels,
    }

    with gzip.open(out_path, "wt", compresslevel=6) as f:
        json.dump(output, f)

    size_mb = out_path.stat().st_size / 1_048_576
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")
    print("\nDone. Use with:")
    print(f"  python -m poker44.miner_model.train --version v6_benchmark --data-source benchmark "
          f"--benchmark-path {out_path}")


if __name__ == "__main__":
    main()
