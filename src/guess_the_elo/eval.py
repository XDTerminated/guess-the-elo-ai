"""Evaluate a trained EloTransformer on the held-out test set.

Loads `checkpoints/best.pt`, runs over `data/processed/test.parquet`,
reports overall MAE + RMSE plus a per-Elo-bucket breakdown so you can
see where the model is strong and where it's weak.

Unlike val MAE, test MAE has *not* been used for checkpoint selection,
so it's the honest unbiased number to report.

Usage:
    uv run python -m guess_the_elo.eval
    uv run python -m guess_the_elo.eval --checkpoint other.pt --test some.parquet
    uv run python -m guess_the_elo.eval --bucket-size 200
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

from guess_the_elo.dataset import make_loader
from guess_the_elo.model import EloTransformer, count_params


class _PortableUnpickler(pickle.Unpickler):
    """Maps Linux `pathlib.PosixPath` -> `pathlib.WindowsPath` at load time.

    Without this, checkpoints saved on Linux (Lightning Studios, etc.) can't
    be loaded on Windows because `PosixPath.__new__` raises there. Done as
    a custom Unpickler rather than monkey-patching pathlib because pickle
    references the class by its real module (`pathlib._local`), and aliasing
    only `pathlib.PosixPath` doesn't reach that lookup path.
    """
    def find_class(self, module: str, name: str):
        if sys.platform == "win32" and name == "PosixPath" and module.startswith("pathlib"):
            return pathlib.WindowsPath
        return super().find_class(module, name)


# torch.load takes a `pickle_module` argument; internally it reads `__name__`
# (to check for dill) plus `Unpickler` / `UnpicklingError` / `HIGHEST_PROTOCOL`.
# A `types.ModuleType` gives us all four with the right names.
_PORTABLE_PICKLE = types.ModuleType("portable_pickle")
_PORTABLE_PICKLE.Unpickler = _PortableUnpickler
_PORTABLE_PICKLE.UnpicklingError = pickle.UnpicklingError
_PORTABLE_PICKLE.HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "best.pt"
DEFAULT_TEST = REPO_ROOT / "data" / "processed" / "test.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--test", type=Path, default=DEFAULT_TEST)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--bucket-size", type=int, default=100,
                   help="Elo bucket width for per-bucket MAE (default: %(default)s).")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def pick_device_and_dtype():
    """Auto-detect: CUDA if available (with the right fp16/bf16), else CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        has_native_bf16 = torch.cuda.get_device_capability() >= (8, 0)
        amp_dtype = torch.bfloat16 if has_native_bf16 else torch.float16
    else:
        device = torch.device("cpu")
        amp_dtype = torch.float32
    return device, amp_dtype


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[EloTransformer, dict]:
    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
        pickle_module=_PORTABLE_PICKLE,
    )
    saved_args = ckpt["args"]
    vocab_size = ckpt["vocab_size"]
    model = EloTransformer(
        vocab_size=vocab_size,
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


@torch.no_grad()
def run_inference(model, loader, device, amp_dtype) -> tuple[np.ndarray, np.ndarray]:
    """Return (preds, targets) each shape [N, 2] over the entire loader."""
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for batch in loader:
        moves = batch["moves"].to(device, non_blocking=True)
        white = batch["white_elo"].to(device, non_blocking=True)
        black = batch["black_elo"].to(device, non_blocking=True)
        target = torch.stack([white, black], dim=1)
        with torch.amp.autocast(device.type, dtype=amp_dtype,
                                enabled=(amp_dtype != torch.float32)):
            pred = model(moves)
        all_preds.append(pred.float().cpu().numpy())
        all_targets.append(target.float().cpu().numpy())
    return np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0)


def print_bucket_breakdown(targets: np.ndarray, abs_errors: np.ndarray, bucket_size: int) -> None:
    """Per-bucket MAE table, keyed on the mean of (white_elo, black_elo)."""
    mean_elo = targets.mean(axis=1)
    buckets = (mean_elo // bucket_size * bucket_size).astype(np.int32)
    per_game_mae = abs_errors.mean(axis=1)

    unique_buckets = np.unique(buckets)
    rows: list[tuple[int, int, float]] = []
    for b in unique_buckets:
        in_b = buckets == b
        rows.append((int(b), int(in_b.sum()), float(per_game_mae[in_b].mean())))

    max_mae = max(r[2] for r in rows)
    print(f"\nPer-bucket MAE  (bucket size {bucket_size}):")
    print(f"  {'Bucket':<13} {'N':>8} {'MAE':>8}")
    print(f"  {'-' * 13} {'-' * 8} {'-' * 8}")
    for b, n, mae in rows:
        bar_len = int(round(40 * mae / max_mae)) if max_mae > 0 else 0
        bar = "#" * bar_len
        print(f"  {b:>5}-{b + bucket_size - 1:<7} {n:>8,} {mae:>8.1f}  {bar}")


def main() -> int:
    args = parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        return 1
    if not args.test.exists():
        print(f"Test set not found: {args.test}")
        return 1

    device, amp_dtype = pick_device_and_dtype()

    model, ckpt = load_model(args.checkpoint, device)
    test_loader = make_loader(
        args.test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        max_len=ckpt["args"]["max_len"],
        pin_memory=(device.type == "cuda"),
    )

    # --- Banner ---
    print(f"Device        : {device}  (AMP dtype: {amp_dtype})")
    print(f"Checkpoint    : {args.checkpoint}  (step {ckpt['step']:,})")
    print(f"  saved val MAE: {ckpt['val_mae']:.1f}")
    print(f"Test games    : {len(test_loader.dataset):,}")
    print(f"Parameters    : {count_params(model):,}")
    print("-" * 60)

    # --- Inference ---
    start = time.time()
    preds, targets = run_inference(model, test_loader, device, amp_dtype)
    elapsed = time.time() - start
    print(f"Inference     : {elapsed:.1f}s  ({len(test_loader.dataset) / elapsed:,.0f} games/sec)")

    # --- Top-line metrics ---
    abs_errors = np.abs(preds - targets)
    overall_mae = float(abs_errors.mean())
    white_mae = float(abs_errors[:, 0].mean())
    black_mae = float(abs_errors[:, 1].mean())
    rmse = float(np.sqrt(((preds - targets) ** 2).mean()))

    print()
    print("=" * 60)
    print(f"  TEST MAE     : {overall_mae:7.1f}  Elo")
    print(f"    White       : {white_mae:7.1f}")
    print(f"    Black       : {black_mae:7.1f}")
    print(f"  TEST RMSE    : {rmse:7.1f}  Elo")
    print("=" * 60)

    # --- Per-bucket breakdown ---
    print_bucket_breakdown(targets, abs_errors, args.bucket_size)

    # --- Sample predictions ---
    print("\nSample predictions (first 10 games):")
    print(f"  {'#':>3}  {'pred W/B':<13}  {'true W/B':<13}  {'avg err':>8}")
    for i in range(min(10, len(preds))):
        pw, pb = preds[i]
        tw, tb = targets[i]
        err = (abs(pw - tw) + abs(pb - tb)) / 2
        print(f"  {i:>3}  {pw:>5.0f} / {pb:<5.0f}  {tw:>5.0f} / {tb:<5.0f}  {err:>8.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
