import math
import chess
import os
import requests
import pandas as pd
import re

# Define the API endpoint and parameters
URL = "https://chess-api.com/v1"

# Define Headers
headers = {"content-type": "application/json"}


def clean_pgn(pgn: str) -> str:
    """
    Remove move numbers (like '1.', '2.', etc.) from the PGN string.

    Args:
        pgn (str): The PGN string containing move numbers.

    Returns:
        str: The cleaned PGN string without move numbers.
    """
    # Use regular expression to remove move numbers (e.g., '1.' or '1...').
    cleaned_pgn = re.sub(r"\d+\.\s*", "", pgn)

    return cleaned_pgn


def remove_columns() -> None:
    """
    Cleans up the data by removing any unnecessary columns.

    Returns:
        None
    """

    # Load the data
    df = pd.read_csv("data/raw/games.csv")

    # Drop the unnecessary columns
    df.drop(
        columns=[
            "id",
            "rated",
            "created_at",
            "last_move_at",
            "increment_code",
            "white_id",
            "black_id",
        ],
        inplace=True,
    )

    # Save the cleaned data
    df.to_csv("data/interim/games.csv", index=False)


def chance_of_winning(centi_pawn_advantage: float) -> float:
    """
    Calculates the chance of winning given the centi pawn advantage.

    Args:
        centi_pawn_advantage (float): Evaluation of the chess position given by stockfish.

    Returns:
        float: The probability of winning

    Source: https://github.com/lichess-org/lila/pull/11148.
    """
    return 50 + 50 * (
        (2 / (math.exp(-0.003682081729595926 * centi_pawn_advantage) + 1)) - 1
    )


def get_centi_pawn_evaluation(fen: str) -> float:
    """
    Get the centi pawn evaluation of a given FEN string.

    Args:
        fen (str): FEN string of the chess position.

    Returns:
        float: The centi pawn evaluation of the chess position.
    """
    payload = {"fen": fen}
    response = requests.post(URL, headers=headers, json=payload)

    return float(response.json()["centipawns"])


def classify_move(chance_of_winning_0: float, chance_of_winning: float) -> int:
    """
    Classify the move as either a blunder (0), mistake (1), inaccuracy (2) or ok (3).

    Args:
        chance_of_winning_0 (float): The probability of winning before the move.
        chance_of_winning (float): The probability of winning after the move.

    Returns:
        int: The classification of the move.
    """
    if chance_of_winning_0 - chance_of_winning >= 30:
        return 0

    elif chance_of_winning_0 - chance_of_winning >= 20:
        return 1

    elif chance_of_winning_0 - chance_of_winning >= 10:
        return 2

    else:
        return 3


def convert_pgn_to_numerical_representation(pgn: str) -> int:
    """
    Converts a PGN string to a numerical representation.

    Args:
        pgn (str): PGN string of the chess game.

    Returns:
        int: Numerical representation of the PGN string.
    """
    pass


def classify_opening(pgn: str) -> str:
    """
    Classify the opening of the chess game.

    Args:
        pgn (str): PGN string of the chess game.

    Returns:
        str: The classification of the opening using ECO.
    """
    pass


def calculate_opening_ply(pgn: str) -> int:
    """
    Calculates the opening ply of the chess game.

    Args:
        pgn (str): PGN string of the chess game.

    Returns:
        int: The opening ply of the chess game.
    """
    pass


def calculate_average_centipawn_loss(pgn: str) -> tuple[float, float]:
    """
    Calculate the average centipawn loss for White and Black in a chess game.

    Args:
        pgn (str): PGN string of the chess game.

    Returns:
        tuple[float, float]: (Average CPL White, Average CPL Black)
    """
    cleaned_pgn = clean_pgn(pgn)
    board = chess.Board()
    moves = cleaned_pgn.split()

    centi_pawn_loss_white = 0.0
    centi_pawn_loss_black = 0.0
    white_move_count = 0
    black_move_count = 0

    for move in moves:
        current_fen = board.fen()
        best_move_evaluation = get_centi_pawn_evaluation(current_fen)

        try:
            board.push_san(move)
        except ValueError as e:
            print(f"Invalid move '{move}' encountered: {e}")
            break  # Exit the loop or handle the invalid move as needed

        next_fen = board.fen()
        actual_move_evaluation = get_centi_pawn_evaluation(next_fen)

        # Determine which player made the move
        if board.turn == chess.BLACK:
            # Last move was by White
            cpl = best_move_evaluation - actual_move_evaluation
            centi_pawn_loss_white += cpl
            white_move_count += 1
        else:
            # Last move was by Black
            cpl = abs(best_move_evaluation - actual_move_evaluation)
            centi_pawn_loss_black += cpl
            black_move_count += 1

    # Calculate average CPL, avoiding division by zero
    average_cpl_white = (
        centi_pawn_loss_white / white_move_count if white_move_count else 0.0
    )
    average_cpl_black = (
        centi_pawn_loss_black / black_move_count if black_move_count else 0.0
    )

    return average_cpl_white, average_cpl_black


def calculate_average_material_imbalance(pgn: str) -> float:  # White - Black
    """
    Calculate the average material imbalance of the chess game.

    Args:
        pgn (str): PGN string of the chess game.

    Returns:
        float: The average material imbalance of the chess game.
    """
    pass
