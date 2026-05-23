"""Split the balanced parquet into stratified train / val / test parquets.

Reads `data/processed/games_balanced.parquet` and writes three files:
  * `data/processed/train.parquet`
  * `data/processed/val.parquet`
  * `data/processed/test.parquet`

The split is stratified by rating bucket (same buckets as 05_balance.py),
so each output preserves the balanced rating distribution.

Usage (from repo root):
    python preprocess/06_split.py
    python preprocess/06_split.py --train-frac 0.8 --val-frac 0.1
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
DEFAULT_SOURCE = REPO_ROOT / "data" / "processed" / "games_balanced.parquet"
DEFAULT_TRAIN = REPO_ROOT / "data" / "processed" / "train.parquet"
DEFAULT_VAL = REPO_ROOT / "data" / "processed" / "val.parquet"
DEFAULT_TEST = REPO_ROOT / "data" / "processed" / "test.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--val-output", type=Path, default=DEFAULT_VAL)
    p.add_argument("--test-output", type=Path, default=DEFAULT_TEST)
    p.add_argument("--train-frac", type=float, default=0.90)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--bucket-size", type=int, default=100,
                   help="Elo bucket width for stratification (default: %(default)s).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.source.exists():
        print(f"Source not found: {args.source}", file=sys.stderr)
        return 1
    if args.train_frac + args.val_frac >= 1.0:
        print("train_frac + val_frac must be < 1.0 (test gets the remainder).",
              file=sys.stderr)
        return 1
    test_frac = 1.0 - args.train_frac - args.val_frac

    for out_path in (args.train_output, args.val_output, args.test_output):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()

    # --- Pass 1: read elo columns, compute buckets, decide split indices ---
    print(f"Reading rating columns from {args.source}...")
    elo_table = pq.read_table(args.source, columns=["white_elo", "black_elo"])
    white = elo_table.column("white_elo").to_numpy().astype(np.int32)
    black = elo_table.column("black_elo").to_numpy().astype(np.int32)
    n_total = white.size
    print(f"Loaded {n_total:,} games")

    mean_elo = (white + black) // 2
    buckets = (mean_elo // args.bucket_size) * args.bucket_size

    rng = np.random.default_rng(args.seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for b in np.unique(buckets):
        in_bucket = np.flatnonzero(buckets == b)
        rng.shuffle(in_bucket)
        n = in_bucket.size
        n_train = int(round(n * args.train_frac))
        n_val = int(round(n * args.val_frac))
        train_parts.append(in_bucket[:n_train])
        val_parts.append(in_bucket[n_train:n_train + n_val])
        test_parts.append(in_bucket[n_train + n_val:])

    train_idx = np.concatenate(train_parts); rng.shuffle(train_idx)
    val_idx = np.concatenate(val_parts);   rng.shuffle(val_idx)
    test_idx = np.concatenate(test_parts);  rng.shuffle(test_idx)

    print(f"\nSplit sizes:")
    print(f"  train : {train_idx.size:>10,}  ({train_idx.size / n_total:.2%})")
    print(f"  val   : {val_idx.size:>10,}  ({val_idx.size / n_total:.2%})")
    print(f"  test  : {test_idx.size:>10,}  ({test_idx.size / n_total:.2%})")

    # --- Pass 2: read full table, take indices, write three parquets ---
    print(f"\nReading full parquet...")
    full = pq.read_table(args.source)

    for name, idx, path in [
        ("train", train_idx, args.train_output),
        ("val",   val_idx,   args.val_output),
        ("test",  test_idx,  args.test_output),
    ]:
        sub = full.take(pa.array(idx))
        pq.write_table(sub, path, compression="zstd")
        size_mb = path.stat().st_size / 1e6
        print(f"  {name:>5} -> {path}  ({size_mb:.1f} MB)")

    elapsed = time.time() - start
    print(f"\nElapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
