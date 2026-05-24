"""FastAPI server exposing the trained EloTransformer over HTTP.

The website (guess-the-elo-website) calls `/predict` with a PGN and a
platform; this server tokenizes, runs inference once, and returns the
predicted Elos.

Run from the repo root:
    uv run uvicorn server:app --port 8000
"""

from __future__ import annotations

import pathlib
import pickle
import sys
import types
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from guess_the_elo.model import EloTransformer
from guess_the_elo.rating import lichess_to_chesscom
from guess_the_elo.tokenizer import Tokenizer


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "best.pt"
DEFAULT_VOCAB = REPO_ROOT / "data" / "processed" / "vocab.json"


# --- Cross-platform pickle (same trick as main.py / eval.py) ---
class _PortableUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if sys.platform == "win32" and name == "PosixPath" and module.startswith("pathlib"):
            return pathlib.WindowsPath
        return super().find_class(module, name)


_PORTABLE_PICKLE = types.ModuleType("portable_pickle")
_PORTABLE_PICKLE.Unpickler = _PortableUnpickler
_PORTABLE_PICKLE.UnpicklingError = pickle.UnpicklingError
_PORTABLE_PICKLE.HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL


# --- Model state, populated once at startup ---
class _State:
    model: EloTransformer | None = None
    tokenizer: Tokenizer | None = None
    device: torch.device | None = None
    max_len: int = 256


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not DEFAULT_CHECKPOINT.exists():
        raise RuntimeError(f"Checkpoint not found: {DEFAULT_CHECKPOINT}")
    if not DEFAULT_VOCAB.exists():
        raise RuntimeError(f"Vocab not found: {DEFAULT_VOCAB}")

    _State.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _State.tokenizer = Tokenizer(DEFAULT_VOCAB)

    ckpt = torch.load(
        DEFAULT_CHECKPOINT,
        map_location=_State.device,
        weights_only=False,
        pickle_module=_PORTABLE_PICKLE,
    )
    saved_args = ckpt["args"]
    _State.max_len = saved_args["max_len"]
    _State.model = EloTransformer(
        vocab_size=ckpt["vocab_size"],
        d_model=saved_args["d_model"],
        n_heads=saved_args["n_heads"],
        n_layers=saved_args["n_layers"],
        d_ff=saved_args["d_ff"],
        max_len=saved_args["max_len"],
        dropout=saved_args["dropout"],
    ).to(_State.device)
    _State.model.load_state_dict(ckpt["model"])
    _State.model.eval()

    print(f"Loaded checkpoint from {DEFAULT_CHECKPOINT}")
    print(f"Device: {_State.device}  Max length: {_State.max_len}  "
          f"Vocab: {_State.tokenizer.vocab_size:,}")
    yield


app = FastAPI(lifespan=lifespan, title="guess-the-elo")

# Allow the Next.js dev server to call us directly during local dev.
# (In production this should be tightened to specific origins.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    pgn: str = Field(..., min_length=1, description="PGN text — with or without headers.")
    platform: Literal["lichess", "chesscom"] = "lichess"


class PredictResponse(BaseModel):
    white_elo: float
    black_elo: float
    white_elo_lichess: float
    black_elo_lichess: float
    platform: Literal["lichess", "chesscom"]
    total_moves: int
    unknown_moves: int
    truncated: bool


@app.get("/health")
def health() -> dict:
    return {"ok": _State.model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _State.model is None or _State.tokenizer is None or _State.device is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    ids = _State.tokenizer.encode(req.pgn)
    if not ids:
        raise HTTPException(
            status_code=400,
            detail="No recognizable SAN moves found in PGN.",
        )
    truncated = False
    if len(ids) > _State.max_len:
        ids = ids[: _State.max_len]
        truncated = True

    moves = torch.tensor([ids], dtype=torch.long, device=_State.device)
    with torch.no_grad():
        pred = _State.model(moves)[0]
    white_lichess, black_lichess = (float(pred[0].item()), float(pred[1].item()))

    if req.platform == "chesscom":
        white_out = lichess_to_chesscom(white_lichess)
        black_out = lichess_to_chesscom(black_lichess)
    else:
        white_out = white_lichess
        black_out = black_lichess

    return PredictResponse(
        white_elo=white_out,
        black_elo=black_out,
        white_elo_lichess=white_lichess,
        black_elo_lichess=black_lichess,
        platform=req.platform,
        total_moves=_State.tokenizer.last_total_moves,
        unknown_moves=_State.tokenizer.last_unknown_count,
        truncated=truncated,
    )
