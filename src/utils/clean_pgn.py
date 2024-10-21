import re


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
