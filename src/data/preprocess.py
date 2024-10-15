import math
import chess
import os
import requests
from dotenv import load_dotenv
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Get the API token from environment variables
API_TOKEN = os.getenv("LICHESS_API_TOKEN")

# Define the API endpoint and parameters
URL = os.getenv("LICHESS_CLOUD_ENGINE")

# Define the headers
headers = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}


def get_fen_representation(position: chess.Board) -> str:
    """
    Get the FEN representation of the chess position.

    Args:
        position (chess.Board): Chess position.

    Returns:
        str: FEN representation of the chess position.
    """
    return position.fen()


def clean_up_data() -> None:
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

    for i in range(len(df)):
        game = df["moves"].iloc[i].split()
        board = chess.Board()

        for j in range(len(game)):
            board.push_san(game[j])
            print(board.fen())

    # Save the cleaned data
    df.to_csv("data/processed/games.csv", index=False)


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
    params = {"fen": fen}
    response = requests.get(URL, headers=headers, params=params)

    return response.json()["pvs"][0]["cp"]


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


def calculate_average_centipawn_loss(pgn: str) -> float:
    """
    Calculate the average centipawn loss of the chess game.

    Args:
        pgn (str): PGN string of the chess game.

    Returns:
        float: The average centipawn loss of the chess game.
    """
    pass


def calculate_average_material_imbalance(pgn: str) -> float:  # White - Black
    """
    Calculate the average material imbalance of the chess game.

    Args:
        pgn (str): PGN string of the chess game.

    Returns:
        float: The average material imbalance of the chess game.
    """
    pass


clean_up_data()
