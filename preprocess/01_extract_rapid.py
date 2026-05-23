"""Extract rated Rapid games from a Lichess monthly PGN dump (.pgn.zst).

Streams the compressed file directly — no full decompression to disk.
A game is kept iff:
  * its `Event` header contains "Rapid", AND
  * both `WhiteElo` and `BlackElo` are present and numeric.

Output is a single uncompressed .pgn file containing the matching games
verbatim (headers + moves), suitable for downstream tokenization.

Usage (from repo root):
    python preprocess/01_extract_rapid.py
    python preprocess/01_extract_rapid.py --delete-source
    python preprocess/01_extract_rapid.py --limit 1000          # quick smoke test
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
from pathlib import Path
from typing import Iterator

import zstandard as zstd
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "data" / "raw" / "lichess_db_standard_rated_2026-04.pgn.zst"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "raw" / "rapid_2026-04.pgn"

HEADER_RE = re.compile(r'\[(\w+)\s+"([^"]*)"\]')


def iter_raw_games(text_stream: io.TextIOBase) -> Iterator[str]:
    """Yield raw PGN text, one game at a time.

    A new game starts when a header line `[...]` appears after we've already
    seen at least one non-empty, non-header line (the moves block).
    """
    buf: list[str] = []
    seen_moves = False
    for line in text_stream:
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


def extract_headers(pgn_text: str) -> dict[str, str]:
    """Parse the header block of a PGN game into a dict. Stops at the
    first non-header line so we don't waste time scanning the moves."""
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


def is_rapid_rated(headers: dict[str, str]) -> bool:
    if "Rapid" not in headers.get("Event", ""):
        return False
    we = headers.get("WhiteElo", "")
    be = headers.get("BlackElo", "")
    return we.isdigit() and be.isdigit()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                   help="Path to the .pgn.zst dump (default: %(default)s).")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help="Path to write filtered rapid games (default: %(default)s).")
    p.add_argument("--delete-source", action="store_true",
                   help="Delete the source .pgn.zst after successful extraction.")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after writing this many games (useful for smoke tests).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.source.exists():
        print(f"Source not found: {args.source}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_size_gb = args.source.stat().st_size / 1e9

    n_total = 0
    n_kept = 0
    start = time.time()

    # Lichess dumps use a large zstd window; allow it explicitly.
    dctx = zstd.ZstdDecompressor(max_window_size=2**31)

    with open(args.source, "rb") as fh, \
         open(args.output, "w", encoding="utf-8", newline="\n") as out_fh:
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            pbar = tqdm(
                unit=" games",
                desc="scanning",
                smoothing=0.05,
                mininterval=0.5,
            )
            for pgn_text in iter_raw_games(text_stream):
                n_total += 1
                pbar.update(1)
                headers = extract_headers(pgn_text)
                if not is_rapid_rated(headers):
                    continue
                out_fh.write(pgn_text)
                if not pgn_text.endswith("\n\n"):
                    out_fh.write("\n")
                n_kept += 1
                if n_kept % 5000 == 0:
                    pbar.set_postfix(kept=n_kept)
                if args.limit is not None and n_kept >= args.limit:
                    break
            pbar.close()

    elapsed = time.time() - start
    out_size_gb = args.output.stat().st_size / 1e9
    print()
    print(f"Scanned : {n_total:,} games")
    print(f"Kept    : {n_kept:,} rated rapid games")
    print(f"Output  : {args.output}  ({out_size_gb:.2f} GB)")
    print(f"Elapsed : {elapsed/60:.1f} min")

    if n_kept == 0:
        print("\nNo games kept — refusing to delete source.", file=sys.stderr)
        return 2

    if args.delete_source:
        print(f"\nDeleting source: {args.source}  ({source_size_gb:.2f} GB)")
        args.source.unlink()
        print("Deleted.")
    else:
        print(f"\nSource kept at {args.source}.")
        print("Re-run with --delete-source (or delete manually) to free disk space.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
