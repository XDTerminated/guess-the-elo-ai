"""Tokenize cleaned rapid PGNs into integer arrays for model training.

Reads the output of `03_clean.py` and writes:
  * `data/processed/games.parquet` — one row per game with white_elo,
    black_elo, and a list of token IDs for the SAN moves.
  * `data/processed/vocab.json` — mapping from SAN move strings to
    integer IDs. ID 0 is reserved for the PAD token.

Tokenization is move-level: each unique SAN string (e.g. "e4", "Nf3",
"O-O", "Bxd5+", "exd5", "e8=Q+", "Qh4#") becomes a distinct token.
Vocab is built on the fly in a single pass — first occurrence of a
move gets the next available ID.

Usage (from repo root):
    python preprocess/04_tokenize.py
    python preprocess/04_tokenize.py --batch-size 20000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "data" / "raw" / "rapid_2026-04_clean.pgn"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "games.parquet"
DEFAULT_VOCAB = REPO_ROOT / "data" / "processed" / "vocab.json"

HEADER_RE = re.compile(r'\[(\w+)\s+"([^"]*)"\]')
COMMENT_RE = re.compile(r"\{[^}]*\}")
MOVE_NUM_RE = re.compile(r"\d+\.+\s*")
ANNOTATION_RE = re.compile(r"[?!]+$")  # ?, !, ?!, !?, ??, !!, ...
NAG_RE = re.compile(r"^\$\d+$")        # $1, $2, ..., $255
RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}

PAD_TOKEN = "<pad>"
UINT16_MAX = 65535


def iter_raw_games(line_iter) -> Iterator[str]:
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


def split_moves_text(pgn_text: str) -> str:
    out: list[str] = []
    in_moves = False
    for line in pgn_text.splitlines():
        if in_moves:
            out.append(line)
        elif not line.startswith("[") and not line.strip():
            in_moves = True
    return "\n".join(out)


def extract_san_moves(pgn_text: str) -> list[str]:
    """Return the SAN move strings from a Lichess PGN game.

    Normalizes annotated moves (e.g. `Nf3?`, `e4!`, `Bxd5?!`) to their
    bare SAN form and drops Numeric Annotation Glyphs ($1, $2, ...).
    These never appear in inference inputs, so collapsing them avoids
    splitting the vocabulary."""
    moves_text = split_moves_text(pgn_text)
    clean = COMMENT_RE.sub("", moves_text)
    clean = MOVE_NUM_RE.sub("", clean)
    out: list[str] = []
    for tok in clean.split():
        if tok in RESULT_TOKENS:
            continue
        if NAG_RE.match(tok):
            continue
        tok = ANNOTATION_RE.sub("", tok)
        if tok:
            out.append(tok)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                   help="Input PGN file (default: %(default)s).")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help="Output Parquet file (default: %(default)s).")
    p.add_argument("--vocab-output", type=Path, default=DEFAULT_VOCAB,
                   help="Output vocab JSON (default: %(default)s).")
    p.add_argument("--batch-size", type=int, default=10000,
                   help="Games per parquet row group (default: %(default)s).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.source.exists():
        print(f"Source not found: {args.source}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.vocab_output.parent.mkdir(parents=True, exist_ok=True)
    source_size = args.source.stat().st_size

    # Vocab grows on the fly. ID 0 reserved for padding.
    vocab: dict[str, int] = {PAD_TOKEN: 0}

    def get_id(token: str) -> int:
        i = vocab.get(token)
        if i is None:
            i = len(vocab)
            vocab[token] = i
        return i

    schema = pa.schema([
        ("white_elo", pa.int16()),
        ("black_elo", pa.int16()),
        ("moves", pa.list_(pa.uint16())),
    ])

    n_total = 0
    n_kept = 0
    n_skipped_empty = 0
    start = time.time()

    pbar = tqdm(
        total=source_size,
        unit="B",
        unit_scale=True,
        desc="tokenizing",
        mininterval=0.5,
    )

    batch_white: list[int] = []
    batch_black: list[int] = []
    batch_moves: list[list[int]] = []

    def flush(writer: pq.ParquetWriter) -> None:
        if not batch_white:
            return
        batch = pa.RecordBatch.from_arrays(
            [
                pa.array(batch_white, type=pa.int16()),
                pa.array(batch_black, type=pa.int16()),
                pa.array(batch_moves, type=pa.list_(pa.uint16())),
            ],
            schema=schema,
        )
        writer.write_batch(batch)
        batch_white.clear()
        batch_black.clear()
        batch_moves.clear()

    with open(args.source, "rb") as in_fh, \
         pq.ParquetWriter(args.output, schema, compression="zstd") as writer:
        for pgn_text in iter_raw_games(iter_decoded_lines(in_fh, pbar)):
            n_total += 1
            headers = extract_headers(pgn_text)
            we, be = headers.get("WhiteElo", ""), headers.get("BlackElo", "")
            if not (we.isdigit() and be.isdigit()):
                n_skipped_empty += 1
                continue
            sans = extract_san_moves(pgn_text)
            if not sans:
                n_skipped_empty += 1
                continue
            ids = [get_id(s) for s in sans]
            batch_white.append(int(we))
            batch_black.append(int(be))
            batch_moves.append(ids)
            n_kept += 1
            if len(batch_white) >= args.batch_size:
                flush(writer)
                pbar.set_postfix(kept=n_kept, vocab=len(vocab))
        flush(writer)
    pbar.close()

    if len(vocab) > UINT16_MAX + 1:
        print(f"WARNING: vocab size ({len(vocab):,}) exceeds uint16 range. "
              "Widen `moves` element type to uint32 before retraining.",
              file=sys.stderr)

    with open(args.vocab_output, "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, indent=2, ensure_ascii=False)

    elapsed = time.time() - start
    out_size_gb = args.output.stat().st_size / 1e9
    print()
    print(f"Scanned : {n_total:,} games")
    print(f"Kept    : {n_kept:,} games")
    print(f"Skipped : {n_skipped_empty:,} games (empty/missing elo)")
    print(f"Vocab   : {len(vocab):,} tokens (incl. PAD)")
    print(f"Output  : {args.output}  ({out_size_gb:.3f} GB)")
    print(f"Vocab   : {args.vocab_output}")
    print(f"Elapsed : {elapsed/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
