"""Balance the rating distribution by capping games per Elo bucket.

Reads `data/processed/games.parquet` and writes a new parquet where each
Elo bucket (default 100-wide, keyed on the mean of `white_elo` and
`black_elo`) contains at most `--max-per-bucket` games. Buckets with
fewer than `--min-per-bucket` games are dropped entirely.

Without this, the model collapses to predicting the mode of the
distribution (~1600 for Lichess rapid). Balancing is typically the
single biggest accuracy lever for Elo prediction.

Usage (from repo root):
    python preprocess/05_balance.py
    python preprocess/05_balance.py --max-per-bucket 50000
    python preprocess/05_balance.py --bucket-size 200 --min-per-bucket 1000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "data" / "processed" / "games.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "games_balanced.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--bucket-size", type=int, default=100,
                   help="Elo bucket width (default: %(default)s).")
    p.add_argument("--max-per-bucket", type=int, default=30000,
                   help="Max games per bucket after balancing (default: %(default)s).")
    p.add_argument("--min-per-bucket", type=int, default=500,
                   help="Drop buckets with fewer than this many games (default: %(default)s, 0 disables).")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducible subsampling (default: %(default)s).")
    return p.parse_args()


def print_histogram(buckets: np.ndarray, mask: np.ndarray, bucket_size: int, title: str) -> None:
    """Print a text histogram of bucket counts under `mask`."""
    sel = buckets[mask] if mask is not None else buckets
    unique, counts = np.unique(sel, return_counts=True)
    if len(counts) == 0:
        print(f"\n{title}: (empty)")
        return
    bar_max = counts.max()
    print(f"\n{title}:  ({sel.size:,} games)")
    for b, c in zip(unique, counts):
        bar_len = int(round(c / bar_max * 50))
        bar = "#" * bar_len
        print(f"  {int(b):>5}-{int(b)+bucket_size-1:<5}: {int(c):>8,}  {bar}")


def main() -> int:
    args = parse_args()

    if not args.source.exists():
        print(f"Source not found: {args.source}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.source.resolve() == args.output.resolve():
        print("Source and output must differ.", file=sys.stderr)
        return 1

    start = time.time()

    # --- Pass 1: read only the rating columns to determine which rows to keep ---
    print(f"Reading rating columns from {args.source}...")
    elo_table = pq.read_table(args.source, columns=["white_elo", "black_elo"])
    white = elo_table.column("white_elo").to_numpy().astype(np.int32)
    black = elo_table.column("black_elo").to_numpy().astype(np.int32)
    n_total = white.size
    print(f"Loaded {n_total:,} games")

    mean_elo = (white + black) // 2
    buckets = (mean_elo // args.bucket_size) * args.bucket_size

    print_histogram(buckets, None, args.bucket_size, "Before balancing")

    # --- Decide which row indices to keep ---
    rng = np.random.default_rng(args.seed)
    unique_buckets, counts = np.unique(buckets, return_counts=True)

    keep_mask = np.zeros(n_total, dtype=bool)
    n_dropped_sparse = 0
    for b, c in zip(unique_buckets, counts):
        if c < args.min_per_bucket:
            n_dropped_sparse += c
            continue
        in_bucket = np.flatnonzero(buckets == b)
        if c > args.max_per_bucket:
            chosen = rng.choice(in_bucket, size=args.max_per_bucket, replace=False)
        else:
            chosen = in_bucket
        keep_mask[chosen] = True

    keep_idx = np.flatnonzero(keep_mask)
    if keep_idx.size == 0:
        print("No games kept — check --min-per-bucket / --max-per-bucket.", file=sys.stderr)
        return 2

    # Shuffle the final order so buckets aren't clumped together.
    rng.shuffle(keep_idx)

    print_histogram(buckets, keep_mask, args.bucket_size, "After balancing")

    # --- Pass 2: read full table, take selected rows, write back ---
    print(f"\nReading full parquet and selecting {keep_idx.size:,} rows...")
    full_table = pq.read_table(args.source)
    filtered = full_table.take(pa.array(keep_idx))
    pq.write_table(filtered, args.output, compression="zstd")

    elapsed = time.time() - start
    n_kept = keep_idx.size
    out_size_mb = args.output.stat().st_size / 1e6

    print()
    print(f"Input         : {n_total:,} games")
    print(f"Dropped sparse: {n_dropped_sparse:,} games")
    print(f"Output        : {n_kept:,} games ({n_kept / n_total:.1%} of input)")
    print(f"File          : {args.output}  ({out_size_mb:.1f} MB)")
    print(f"Elapsed       : {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
