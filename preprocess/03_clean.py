"""Clean PGN games by dropping bots, abnormal terminations, and mismatched pairings.

Reads the output of `02_filter_length.py` and writes a new PGN excluding:
  * Bot games — either side has `[WhiteTitle "BOT"]` or `[BlackTitle "BOT"]`.
  * Abnormal terminations — `Termination` in {Abandoned, Rules infraction, Unterminated}.
  * Mismatched pairings — `|WhiteElo - BlackElo| > --max-rating-diff` (default 400).

All three are cheap header-only checks done in a single pass.

Usage (from repo root):
    python preprocess/03_clean.py
    python preprocess/03_clean.py --max-rating-diff 300
    python preprocess/03_clean.py --delete-source
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Iterator

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "data" / "raw" / "rapid_2026-04_filtered.pgn"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "raw" / "rapid_2026-04_clean.pgn"

HEADER_RE = re.compile(r'\[(\w+)\s+"([^"]*)"\]')
BAD_TERMINATIONS = {"Abandoned", "Rules infraction", "Unterminated"}


def iter_raw_games(line_iter) -> Iterator[str]:
    """Yield raw PGN text, one game at a time. A new game starts when a
    header line appears after we've already seen the moves block."""
    buf: list[str] = []
    seen_moves = False
    for line in line_iter:
        is_header = line.startswith("[")
        if is_header and seen_moves:
            yield "".join(buf)
            buf = [line]
            seen_moves = False
            continue
        buf.append(line)
        if not is_header and line.strip():
            seen_moves = True
    if buf:
        yield "".join(buf)


def iter_decoded_lines(binary_fh, pbar) -> Iterator[str]:
    """Yield text lines from a binary file handle, updating `pbar` with
    the number of bytes consumed per line. Done this way because Python
    disables `tell()` on a text-mode file once it has been iterated."""
    for raw_line in binary_fh:
        pbar.update(len(raw_line))
        yield raw_line.decode("utf-8", errors="replace")


def extract_headers(pgn_text: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in pgn_text.splitlines():
        if not line.startswith("["):
            if headers:
                break
            continue
        m = HEADER_RE.match(line)
        if m:
            headers[m.group(1)] = m.group(2)
    return headers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                   help="Input PGN file (default: %(default)s).")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help="Output PGN file (default: %(default)s).")
    p.add_argument("--max-rating-diff", type=int, default=400,
                   help="Drop games where |WhiteElo - BlackElo| exceeds this (default: %(default)s).")
    p.add_argument("--delete-source", action="store_true",
                   help="Delete the input PGN after a successful run.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.source.exists():
        print(f"Source not found: {args.source}", file=sys.stderr)
        return 1
    if args.source.resolve() == args.output.resolve():
        print("Source and output must differ.", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_size = args.source.stat().st_size

    n_total = 0
    n_kept = 0
    n_drop_bot = 0
    n_drop_termination = 0
    n_drop_rating_diff = 0
    start = time.time()

    pbar = tqdm(
        total=source_size,
        unit="B",
        unit_scale=True,
        desc="cleaning",
        mininterval=0.5,
    )
    with open(args.source, "rb") as in_fh, \
         open(args.output, "w", encoding="utf-8", newline="\n") as out_fh:
        for pgn_text in iter_raw_games(iter_decoded_lines(in_fh, pbar)):
            n_total += 1

            headers = extract_headers(pgn_text)

            # (1) Drop bot games.
            if headers.get("WhiteTitle") == "BOT" or headers.get("BlackTitle") == "BOT":
                n_drop_bot += 1
                continue

            # (2) Drop abnormal terminations.
            if headers.get("Termination", "Normal") in BAD_TERMINATIONS:
                n_drop_termination += 1
                continue

            # (3) Drop extreme rating mismatches.
            we = headers.get("WhiteElo", "")
            be = headers.get("BlackElo", "")
            if we.isdigit() and be.isdigit():
                if abs(int(we) - int(be)) > args.max_rating_diff:
                    n_drop_rating_diff += 1
                    continue
            else:
                # No usable Elo (shouldn't happen after step 1, but be defensive).
                n_drop_rating_diff += 1
                continue

            out_fh.write(pgn_text)
            if not pgn_text.endswith("\n\n"):
                out_fh.write("\n")
            n_kept += 1
            if n_kept % 5000 == 0:
                pbar.set_postfix(kept=n_kept, scanned=n_total)
    pbar.close()

    elapsed = time.time() - start
    out_size_gb = args.output.stat().st_size / 1e9

    print()
    print(f"Scanned          : {n_total:,} games")
    print(f"Dropped (bot)    : {n_drop_bot:,}")
    print(f"Dropped (term.)  : {n_drop_termination:,}")
    print(f"Dropped (rating) : {n_drop_rating_diff:,}  (|diff| > {args.max_rating_diff})")
    print(f"Kept             : {n_kept:,} games")
    print(f"Output           : {args.output}  ({out_size_gb:.2f} GB)")
    print(f"Elapsed          : {elapsed/60:.1f} min")

    if n_kept == 0:
        print("\nNo games kept — refusing to delete source.", file=sys.stderr)
        return 2

    if args.delete_source:
        print(f"\nDeleting source: {args.source}  ({source_size / 1e9:.2f} GB)")
        args.source.unlink()
        print("Deleted.")
    else:
        print(f"\nSource kept at {args.source}.")
        print("Re-run with --delete-source (or delete manually) to free disk space.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
