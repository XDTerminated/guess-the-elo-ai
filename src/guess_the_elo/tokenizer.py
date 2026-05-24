"""SAN move tokenizer for runtime PGN inputs.

Loads `data/processed/vocab.json` (produced by `preprocess/04_tokenize.py`)
and exposes a `Tokenizer` class that converts a raw PGN string into a
list of integer token IDs the EloTransformer can consume.

The normalization rules MUST match `04_tokenize.py` exactly — otherwise
training and inference see different tokens for the same move:
  * Strip `{...}` comments (Lichess eval/clock annotations).
  * Strip move-number markers like `1.` and `1...`.
  * Strip trailing `?`/`!` from each SAN move (analysis annotations).
  * Drop Numeric Annotation Glyphs (`$1`, `$2`, ...).
  * Drop game-result tokens (`1-0`, `0-1`, `1/2-1/2`, `*`).

Moves not in the training vocab are silently dropped — no UNK token was
reserved at training time, so they have no embedding to use. The count
of dropped moves is exposed via `Tokenizer.last_unknown_count` so the
caller can warn the user if a large fraction of the game was OOV.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


COMMENT_RE = re.compile(r"\{[^}]*\}")
MOVE_NUM_RE = re.compile(r"\d+\.+\s*")
ANNOTATION_RE = re.compile(r"[?!]+$")
NAG_RE = re.compile(r"^\$\d+$")
RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}


def split_moves_text(pgn_text: str) -> str:
    """Return only the moves portion of a PGN.

    If a header block is present, return everything after the blank line
    that separates headers from moves. If the input has no headers at all
    (e.g. a bare move list pasted in), return the whole input.
    """
    lines = pgn_text.splitlines()
    has_headers = any(line.startswith("[") for line in lines)
    if not has_headers:
        return pgn_text
    out: list[str] = []
    in_moves = False
    for line in lines:
        if in_moves:
            out.append(line)
        elif not line.startswith("[") and not line.strip():
            in_moves = True
    return "\n".join(out)


def extract_san_moves(pgn_text: str) -> list[str]:
    """Pull normalized SAN move tokens out of an arbitrary PGN string."""
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


class Tokenizer:
    """Loads vocab.json once, reusable for many encode calls."""

    def __init__(self, vocab_path: str | Path):
        self.vocab_path = Path(vocab_path)
        with open(self.vocab_path, encoding="utf-8") as fh:
            self.vocab: dict[str, int] = json.load(fh)
        self.id_to_token: list[str] = [""] * len(self.vocab)
        for tok, i in self.vocab.items():
            if 0 <= i < len(self.id_to_token):
                self.id_to_token[i] = tok
        # Per-call diagnostics — useful for warning the caller about OOV.
        self.last_unknown_count: int = 0
        self.last_total_moves: int = 0

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, pgn_text: str) -> list[int]:
        """PGN string -> list of token IDs. Unknown moves are dropped;
        the count is recorded on `self.last_unknown_count`."""
        sans = extract_san_moves(pgn_text)
        ids: list[int] = []
        unknown = 0
        for s in sans:
            i = self.vocab.get(s)
            if i is None:
                unknown += 1
                continue
            ids.append(i)
        self.last_total_moves = len(sans)
        self.last_unknown_count = unknown
        return ids

    def decode(self, ids: list[int]) -> list[str]:
        """Token IDs -> SAN moves. Useful for debugging tokenization."""
        out: list[str] = []
        for i in ids:
            if 0 <= i < len(self.id_to_token):
                out.append(self.id_to_token[i])
        return out
