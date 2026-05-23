"""Transformer model for chess Elo prediction.

Takes a tokenized game (sequence of SAN move IDs) and predicts both
players' Elos as raw scalars.

Architecture:
  * Token + learned positional embeddings.
  * Pre-norm Transformer encoder.
  * Learnable [CLS] token prepended to each sequence; its final-layer
    representation is fed to a 2-layer MLP head producing 2 scalars
    (white_elo, black_elo).

The padding mask is derived from `moves == pad_id`, so callers don't
need to pass lengths explicitly.

Run this file directly to verify shapes and parameter count:
    uv run python src/guess_the_elo/model.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EloTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_len: int = 256,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        # +1 to leave room for the prepended CLS token.
        self.pos_emb = nn.Embedding(max_len + 1, d_model)

        # Learnable [CLS] token; its final-layer embedding represents the whole game.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm — more stable than post-norm at this scale.
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        # 2-layer MLP head — small non-linearity on the pooled embedding.
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, moves: torch.Tensor) -> torch.Tensor:
        """
        Args:
            moves: [B, L] LongTensor of token IDs (padded with `pad_id`).
        Returns:
            [B, 2] FloatTensor of predicted (white_elo, black_elo) in raw Elo units.
        """
        B, L = moves.shape
        assert L + 1 <= self.pos_emb.num_embeddings, (
            f"Sequence too long: {L + 1} > {self.pos_emb.num_embeddings}. "
            f"Increase model `max_len` or dataset truncation."
        )

        x = self.token_emb(moves)  # [B, L, D]

        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls, x], dim=1)  # [B, L+1, D]

        positions = torch.arange(L + 1, device=moves.device).unsqueeze(0)  # [1, L+1]
        x = x + self.pos_emb(positions)

        # `src_key_padding_mask`: True at positions to ignore (i.e. padding).
        move_pad_mask = (moves == self.pad_id)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=moves.device)
        key_padding_mask = torch.cat([cls_mask, move_pad_mask], dim=1)  # [B, L+1]

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        cls_out = self.norm(x[:, 0])  # [B, D]
        return self.head(cls_out)  # [B, 2]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _sanity_check() -> int:
    """Construct the model, push a dummy batch through, print shapes + param count."""
    import json
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    vocab_path = repo_root / "data" / "processed" / "vocab.json"
    if vocab_path.exists():
        with open(vocab_path, encoding="utf-8") as fh:
            vocab = json.load(fh)
        vocab_size = len(vocab)
    else:
        print(f"No vocab at {vocab_path} — using placeholder size 7000.")
        vocab_size = 7000

    model = EloTransformer(vocab_size=vocab_size)
    print(f"Vocab size : {vocab_size:,}")
    print(f"Parameters : {count_params(model):,}")

    # Dummy batch: 4 games, padded to length 50 with some pad sprinkled in.
    moves = torch.randint(1, vocab_size, (4, 50))
    moves[2, 30:] = 0
    moves[3, 10:] = 0

    out = model(moves)
    print(f"Input shape : {tuple(moves.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Sample preds: {[round(v, 2) for v in out[0].tolist()]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_sanity_check())
