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

    return response.json()


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
    Calculate the average centipawn loss of the chess game.

    Args:
        pgn (str): PGN string of the chess game.

    Returns:
        float: The average centipawn loss of the chess game.
    """

    pgn = clean_pgn(pgn)

    move_number = 0

    board = chess.Board()
    pgn = pgn.split()

    centi_pawn_loss_white = 0
    centi_pawn_loss_black = 0

    for move in pgn:
        current_evaluation = get_centi_pawn_evaluation(board.fen())
        board.push_san(move)
        next_evaluation = get_centi_pawn_evaluation(board.fen())

        if move_number % 2 == 0:
            centi_pawn_loss_white += current_evaluation - next_evaluation

        else:
            centi_pawn_loss_black += current_evaluation - next_evaluation

        move_number += 1

    return (centi_pawn_loss_white) / (move / 2), (centi_pawn_loss_black) / (move / 2)


def calculate_average_material_imbalance(pgn: str) -> float:  # White - Black
    """
    Calculate the average material imbalance of the chess game.

    Args:
        pgn (str): PGN string of the chess game.

    Returns:
        float: The average material imbalance of the chess game.
    """
    pass


print(
    get_centi_pawn_evaluation(
        "rnb1kbnr/ppppq1pp/8/4P3/4p3/2N5/PPP2PPP/R1BQKBNR w KQkq - 2 5"
    )
)

# # clean_up_data()
# calculate_average_centipawn_loss(
#     "1. e4 e5 2. d4 f5 3. dxe5 fxe4 4. Nc3 Qe7 5. Nxe4 Qxe5 6. Qe2 Bb4+ 7. c3 Ba5 8. Ng3 Qe7 9. Qxe7+ Nxe7 10. Nf3 d5 11. Bd3 Nbc6 12. Be3 Bd7 13. O-O O-O-O 14. Nd4 Ne5 15. Bc2 h6 16. f4 Nc4 17. b4 Nxe3 18. bxa5 Nxf1 19. Kxf1 a6 20. Rb1 c5 21. Nde2 Bb5 22. a4 Bc4 23. Nf5 Nc6 24. Nxg7 Nxa5 25. Bf5+ Kb8 26. Ne6 Rde8 27. Bg4 Ba2 28. Rb2 Bc4 29. Nxc5 Rxe2 30. Bxe2 Bxe2+ 31. Rxe2 Nc4 32. Re7 Ne3+ 33. Rxe3"
# )
