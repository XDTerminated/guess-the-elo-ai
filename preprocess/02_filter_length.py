"""Filter PGN games by minimum game length.

Reads the output of `01_extract_rapid.py` and writes a new PGN containing
only games whose length is **strictly greater than** `--min-moves` full
moves. A "full move" is one white ply + one black ply (so 10 full moves
means 20 plies); default is 10, matching "more than 10 moves long".

Game length is determined by finding the highest move number marker
(e.g. `15.`) in the moves text after stripping `{...}` comment blocks
(Lichess includes per-move eval/clock annotations there).

Usage (from repo root):
    python preprocess/02_filter_length.py
    python preprocess/02_filter_length.py --min-moves 20
    python preprocess/02_filter_length.py --delete-source
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
DEFAULT_SOURCE = REPO_ROOT / "data" / "raw" / "rapid_2026-04.pgn"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "raw" / "rapid_2026-04_filtered.pgn"

COMMENT_RE = re.compile(r"\{[^}]*\}")
MOVE_NUM_RE = re.compile(r"\b(\d+)\.")


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


def split_moves_text(pgn_text: str) -> str:
    """Return only the moves portion of a PGN (everything after the
    blank line that separates headers from moves)."""
    out: list[str] = []
    in_moves = False
    for line in pgn_text.splitlines():
        if in_moves:
            out.append(line)
        elif not line.startswith("[") and not line.strip():
            in_moves = True
    return "\n".join(out)


def count_full_moves(pgn_text: str) -> int:
    """Return the highest move number reached in the game, which equals
    the number of full moves played."""
    moves = split_moves_text(pgn_text)
    clean = COMMENT_RE.sub("", moves)
    nums = MOVE_NUM_RE.findall(clean)
    return max((int(n) for n in nums), default=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                   help="Input PGN file (default: %(default)s).")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help="Output PGN file (default: %(default)s).")
    p.add_argument("--min-moves", type=int, default=10,
                   help="Keep games STRICTLY LONGER than this many full moves "
                        "(default: %(default)s, i.e. keep games with >= 11 full moves).")
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
    start = time.time()

    pbar = tqdm(
        total=source_size,
        unit="B",
        unit_scale=True,
        desc="filtering",
        mininterval=0.5,
    )
    with open(args.source, "rb") as in_fh, \
         open(args.output, "w", encoding="utf-8", newline="\n") as out_fh:
        for pgn_text in iter_raw_games(iter_decoded_lines(in_fh, pbar)):
            n_total += 1
            if count_full_moves(pgn_text) > args.min_moves:
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
    print(f"Scanned : {n_total:,} games")
    print(f"Kept    : {n_kept:,} games (>{args.min_moves} full moves)")
    print(f"Dropped : {n_total - n_kept:,} games")
    print(f"Output  : {args.output}  ({out_size_gb:.2f} GB)")
    print(f"Elapsed : {elapsed/60:.1f} min")

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
