"""Convert between Lichess and Chess.com ratings.

Uses NoseKnowsAll's "Universal Rating Converter" lookup table (2024):
    https://lichess.org/@/NoseKnowsAll/blog/introducing-a-universal-rating-converter-for-2024/X2QAH27t

The table was built by analyzing thousands of ChessDojo members with
stable ratings across both platforms. It's empirical, not formulaic.

Caveat: the table's "Lichess" column is technically the *classical*
rating, while the Chess.com column is rapid. The model in this project
is trained on Lichess rapid. For most rapid players the Lichess
classical/rapid difference is small (<50 Elo), so we use the table as-is.

Out-of-range inputs are clamped to the nearest endpoint, because
extrapolating off the ends of an empirical table is unreliable.
"""

from __future__ import annotations

import numpy as np


# Lookup table rows from the article, sorted by Lichess rating ascending.
# (The article's "Cohort" column corresponds roughly to Chess.com rapid,
# which is what we put in `_CHESSCOM` below.)
_LICHESS = np.array(
    [
        1250, 1310, 1370, 1435, 1500, 1550, 1600, 1665, 1730, 1795,
        1850, 1910, 1970, 2030, 2090, 2150, 2225, 2310, 2370, 2410,
        2440, 2470,
    ],
    dtype=float,
)

_CHESSCOM = np.array(
    [
         550,  650,  750,  850,  950, 1050, 1150, 1250, 1350, 1450,
        1550, 1650, 1750, 1850, 1950, 2050, 2165, 2275, 2360, 2425,
        2485, 2550,
    ],
    dtype=float,
)


def lichess_to_chesscom(lichess_elo: float) -> float:
    """Convert a Lichess rating to its Chess.com equivalent.

    Inputs below 1250 Lichess clamp to 550 Chess.com; above 2470 clamp to 2550.
    """
    return float(np.interp(lichess_elo, _LICHESS, _CHESSCOM))


def chesscom_to_lichess(chesscom_elo: float) -> float:
    """Convert a Chess.com rating to its Lichess equivalent (inverse of above)."""
    return float(np.interp(chesscom_elo, _CHESSCOM, _LICHESS))
