"""Training loop for the EloTransformer.

Trains on `data/processed/train.parquet`, validates on `val.parquet`,
saves the best checkpoint (by val MAE) to `checkpoints/best.pt`.

Loss is Huber (smooth L1) on Elo values normalized as
    (elo - ELO_MEAN) / ELO_STD
so gradients stay sane. The model itself still outputs raw Elo;
normalization is loss-side only.

Run from repo root:
    uv run python -m guess_the_elo.train
    uv run python -m guess_the_elo.train --epochs 10 --batch-size 128
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from guess_the_elo.dataset import DEFAULT_MAX_LEN, make_loader
from guess_the_elo.model import EloTransformer, count_params


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = REPO_ROOT / "data" / "processed" / "train.parquet"
DEFAULT_VAL = REPO_ROOT / "data" / "processed" / "val.parquet"
DEFAULT_VOCAB = REPO_ROOT / "data" / "processed" / "vocab.json"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "best.pt"

# Loss-side normalization. Picked from the Lichess rapid distribution
# after balancing — roughly uniform in [900, 2400], so mean ~1500 / std ~400.
ELO_MEAN = 1500.0
ELO_STD = 400.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Data
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--val", type=Path, default=DEFAULT_VAL)
    p.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    p.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    # Training
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=500)
    # Model
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    # Misc
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def pick_device_and_dtype():
    """Require CUDA. Fail loudly otherwise — we are paying for a GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available, but this script requires a GPU.\n"
            "Most likely cause: the CPU-only PyTorch wheel is installed.\n"
            "On a GPU box, reinstall PyTorch with the matching CUDA index URL, e.g.:\n"
            "    pip install --index-url https://download.pytorch.org/whl/cu121 torch"
        )
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None
    return device, amp_dtype, scaler


def make_lr_scheduler(optimizer, warmup_steps: int, total_steps: int, min_ratio: float = 0.1):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def huber_loss_normalized(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_n = (pred - ELO_MEAN) / ELO_STD
    target_n = (target - ELO_MEAN) / ELO_STD
    return F.smooth_l1_loss(pred_n, target_n)


@torch.no_grad()
def validate(model, loader, device, amp_dtype) -> dict:
    model.eval()
    total_loss = 0.0
    total_abs = 0.0
    n_games = 0
    for batch in loader:
        moves = batch["moves"].to(device, non_blocking=True)
        white = batch["white_elo"].to(device, non_blocking=True)
        black = batch["black_elo"].to(device, non_blocking=True)
        target = torch.stack([white, black], dim=1)
        with torch.amp.autocast(device.type,
                                dtype=amp_dtype,
                                enabled=(amp_dtype != torch.float32)):
            pred = model(moves)
            loss = huber_loss_normalized(pred, target)
        n = moves.shape[0]
        total_loss += loss.item() * n
        total_abs += (pred.float() - target).abs().sum().item()
        n_games += n
    return {
        "loss": total_loss / n_games,
        "mae": total_abs / (n_games * 2),  # 2 outputs per game
    }


def save_checkpoint(path: Path, model, optimizer, step, epoch, val_mae, args, vocab_size):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "val_mae": val_mae,
        "args": vars(args),
        "vocab_size": vocab_size,
    }, path)


def train(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)

    device, amp_dtype, scaler = pick_device_and_dtype()
    gpu_name = torch.cuda.get_device_name(0)

    # Vocab — needed to size the model.
    if not args.vocab.exists():
        print(f"Vocab not found: {args.vocab}")
        return 1
    with open(args.vocab, encoding="utf-8") as fh:
        vocab = json.load(fh)
    vocab_size = len(vocab)

    # Data loaders.
    train_loader = make_loader(
        args.train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, max_len=args.max_len, pin_memory=True,
    )
    val_loader = make_loader(
        args.val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, max_len=args.max_len, pin_memory=True,
    )

    # Model.
    model = EloTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_len=args.max_len,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = make_lr_scheduler(optimizer, args.warmup_steps, total_steps)

    # --- Run banner ---
    print(f"Device        : {device}  ({gpu_name})")
    print(f"AMP dtype     : {amp_dtype}")
    print(f"Vocab size    : {vocab_size:,}")
    print(f"Train games   : {len(train_loader.dataset):,}")
    print(f"Val games     : {len(val_loader.dataset):,}")
    print(f"Parameters    : {count_params(model):,}")
    print(f"Total steps   : {total_steps:,}  (warmup {args.warmup_steps:,})")
    print(f"Checkpoint    : {args.checkpoint}")
    print("-" * 70)

    best_val_mae = float("inf")
    step = 0
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            moves = batch["moves"].to(device, non_blocking=True)
            white = batch["white_elo"].to(device, non_blocking=True)
            black = batch["black_elo"].to(device, non_blocking=True)
            target = torch.stack([white, black], dim=1)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type,
                                    dtype=amp_dtype,
                                    enabled=(amp_dtype != torch.float32)):
                pred = model(moves)
                loss = huber_loss_normalized(pred, target)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
            scheduler.step()

            step += 1

            if step % args.log_every == 0:
                with torch.no_grad():
                    mae = (pred.detach().float() - target).abs().mean().item()
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start
                ips = step / elapsed
                print(f"step {step:>6}/{total_steps}  ep {epoch+1}/{args.epochs}  "
                      f"lr {lr:.2e}  loss {loss.item():.4f}  mae {mae:6.1f}  "
                      f"{ips:5.1f} it/s")

            if step % args.val_every == 0 or step == total_steps:
                val_metrics = validate(model, val_loader, device, amp_dtype)
                star = ""
                if val_metrics["mae"] < best_val_mae:
                    best_val_mae = val_metrics["mae"]
                    star = "  *new best*"
                    save_checkpoint(args.checkpoint, model, optimizer,
                                    step, epoch, val_metrics["mae"], args, vocab_size)
                print(f"  >> VAL: loss {val_metrics['loss']:.4f}  "
                      f"mae {val_metrics['mae']:6.1f}{star}")
                model.train()

    print("-" * 70)
    print(f"Done. Best val MAE: {best_val_mae:.1f}")
    return 0


def main() -> int:
    args = parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
