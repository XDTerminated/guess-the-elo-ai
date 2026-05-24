"""Predict white/black Elo from a PGN.

Usage:
    python main.py path/to/game.pgn        # PGN file
    python main.py < game.pgn               # stdin
    python main.py --paste                  # interactive paste (end with Ctrl+D / Ctrl+Z+Enter)

Loads the best checkpoint from `checkpoints/best.pt` and the vocab from
`data/processed/vocab.json`. Inference runs on GPU if available, else CPU.
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import sys
import types
from pathlib import Path

import torch

from guess_the_elo.model import EloTransformer
from guess_the_elo.tokenizer import Tokenizer


# Cross-platform pickle so checkpoints saved on Linux load on Windows.
# (Same trick as in eval.py — pickle stores `pathlib.PosixPath`, which
# can't be instantiated on Windows; we redirect it to `WindowsPath`.)
class _PortableUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if sys.platform == "win32" and name == "PosixPath" and module.startswith("pathlib"):
            return pathlib.WindowsPath
        return super().find_class(module, name)


_PORTABLE_PICKLE = types.ModuleType("portable_pickle")
_PORTABLE_PICKLE.Unpickler = _PortableUnpickler
_PORTABLE_PICKLE.UnpicklingError = pickle.UnpicklingError
_PORTABLE_PICKLE.HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "best.pt"
DEFAULT_VOCAB = REPO_ROOT / "data" / "processed" / "vocab.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("pgn", nargs="?", type=Path,
                   help="PGN file to read. If omitted, reads from stdin.")
    p.add_argument("--paste", action="store_true",
                   help="Interactive paste mode (read until EOF).")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                   help="Checkpoint to load (default: %(default)s).")
    p.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB,
                   help="Vocab JSON to load (default: %(default)s).")
    return p.parse_args()


def read_pgn(args: argparse.Namespace) -> str:
    if args.paste:
        print("Paste PGN, then EOF (Ctrl+D on Linux/Mac, Ctrl+Z + Enter on Windows):",
              file=sys.stderr)
        return sys.stdin.read()
    if args.pgn is not None:
        if not args.pgn.exists():
            print(f"Error: file not found: {args.pgn}", file=sys.stderr)
            sys.exit(1)
        return args.pgn.read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    print("Error: no PGN provided. Pass a file path, pipe via stdin, or use --paste.",
          file=sys.stderr)
    sys.exit(1)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[EloTransformer, dict]:
    if not checkpoint_path.exists():
        print(f"Error: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)
    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
        pickle_module=_PORTABLE_PICKLE,
    )
    saved_args = ckpt["args"]
    model = EloTransformer(
        vocab_size=ckpt["vocab_size"],
        d_model=saved_args["d_model"],
        n_heads=saved_args["n_heads"],
        n_layers=saved_args["n_layers"],
        d_ff=saved_args["d_ff"],
        max_len=saved_args["max_len"],
        dropout=saved_args["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def main() -> int:
    args = parse_args()

    pgn_text = read_pgn(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.vocab.exists():
        print(f"Error: vocab not found: {args.vocab}", file=sys.stderr)
        return 1
    tokenizer = Tokenizer(args.vocab)
    model, ckpt = load_model(args.checkpoint, device)
    max_len = ckpt["args"]["max_len"]

    ids = tokenizer.encode(pgn_text)
    if not ids:
        print("Error: no recognizable SAN moves found in input.", file=sys.stderr)
        return 1
    truncated = False
    if len(ids) > max_len:
        ids = ids[:max_len]
        truncated = True

    moves = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        pred = model(moves)[0]
    white_elo, black_elo = pred.tolist()

    print(f"White: {white_elo:>4.0f} Elo")
    print(f"Black: {black_elo:>4.0f} Elo")

    # --- Diagnostics on stderr so they don't pollute clean stdout ---
    notes: list[str] = []
    if tokenizer.last_unknown_count > 0:
        pct = 100 * tokenizer.last_unknown_count / tokenizer.last_total_moves
        notes.append(
            f"{tokenizer.last_unknown_count}/{tokenizer.last_total_moves} moves "
            f"({pct:.1f}%) were not in the training vocab and were skipped"
        )
    if truncated:
        notes.append(f"game truncated to first {max_len} plies for the model")
    for n in notes:
        print(f"(note: {n})", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
