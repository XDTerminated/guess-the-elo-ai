"""PyTorch Dataset + DataLoader for tokenized chess games.

Reads a parquet produced by `06_split.py` (schema: `white_elo: int16`,
`black_elo: int16`, `moves: list<uint16>`) and yields padded batches.

Defaults:
  * `max_len = 256` plies — covers ~99% of rapid games end-to-end; longer
    games are truncated.
  * Dynamic per-batch padding — each batch is padded to *its own* longest
    game (not a global max), so short batches don't waste compute.
  * `PAD_ID = 0` — matches the PAD token reserved by `04_tokenize.py`.

Run this file directly as a sanity check:
    uv run python src/guess_the_elo/dataset.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset


PAD_ID = 0  # must match vocab.json from 04_tokenize.py
DEFAULT_MAX_LEN = 256


class ChessEloDataset(Dataset):
    """Loads a tokenized-games parquet into memory and yields one game at a time.

    Each item is a dict with:
        `moves`     : LongTensor of token IDs, truncated at `max_len`.
        `white_elo` : float (target).
        `black_elo` : float (target).
    """

    def __init__(self, parquet_path: str | Path, max_len: int = DEFAULT_MAX_LEN):
        self.max_len = max_len
        table = pq.read_table(parquet_path)
        # Elos are small ints — numpy is plenty.
        self.white_elo = table.column("white_elo").to_numpy().astype(np.int32)
        self.black_elo = table.column("black_elo").to_numpy().astype(np.int32)
        # `moves` is a list<uint16> ChunkedArray — combining chunks gives O(1) row access.
        self.moves = table.column("moves").combine_chunks()

    def __len__(self) -> int:
        return int(self.white_elo.shape[0])

    def __getitem__(self, idx: int) -> dict:
        m = self.moves[idx].as_py()
        if len(m) > self.max_len:
            m = m[: self.max_len]
        return {
            "moves": torch.tensor(m, dtype=torch.long),
            "white_elo": float(self.white_elo[idx]),
            "black_elo": float(self.black_elo[idx]),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad each batch to its own longest game (dynamic padding)."""
    lengths = torch.tensor([s["moves"].shape[0] for s in batch], dtype=torch.long)
    max_len = int(lengths.max())
    moves = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    for i, s in enumerate(batch):
        n = s["moves"].shape[0]
        moves[i, :n] = s["moves"]
    return {
        "moves": moves,
        "lengths": lengths,
        "white_elo": torch.tensor([s["white_elo"] for s in batch], dtype=torch.float32),
        "black_elo": torch.tensor([s["black_elo"] for s in batch], dtype=torch.float32),
    }


def make_loader(
    parquet_path: str | Path,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    max_len: int = DEFAULT_MAX_LEN,
    pin_memory: bool = True,
) -> DataLoader:
    """Convenience: ChessEloDataset wrapped in a DataLoader with the right collate."""
    ds = ChessEloDataset(parquet_path, max_len=max_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )


def _sanity_check() -> int:
    """Load val.parquet, print one batch's shapes + a sample."""
    repo_root = Path(__file__).resolve().parents[2]
    val_path = repo_root / "data" / "processed" / "val.parquet"
    if not val_path.exists():
        print(f"No file at {val_path}. Run the preprocessing pipeline first.")
        return 1

    loader = make_loader(val_path, batch_size=8, shuffle=False)
    batch = next(iter(loader))

    print(f"Loaded {len(loader.dataset):,} games from {val_path.name}")
    print("Batch shapes:")
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"  {k:>9}: {tuple(v.shape)} {v.dtype}")
    print("\nFirst sample:")
    print(f"  white_elo : {batch['white_elo'][0].item():.0f}")
    print(f"  black_elo : {batch['black_elo'][0].item():.0f}")
    print(f"  length    : {batch['lengths'][0].item()}")
    print(f"  tokens[:10]: {batch['moves'][0][:10].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_sanity_check())
