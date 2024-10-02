import math


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
    pass


def classify_move() -> str:
    """
    Classify the move as either a blunder, mistake, or inaccuracy.

    Returns:
        str: The classification of the move.
    """
    pass


def convert_pgn_to_numerical_representation(pgn: str) -> int:
    """
    Converts a PGN string to a numerical representation.

    Args:
        pgn (str): PGN string of the chess game.

    Returns:
        int: Numerical representation of the PGN string.
    """
    pass
