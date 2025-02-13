import re


class MoveCounter:
    def __init__(self):
        self.comment_pattern = re.compile(r"\{.*?\}")
        self.variation_pattern = re.compile(r"\(.*?\)")
        self.move_number_pattern = re.compile(r"^\d+\.?$")

    def count_plies(self, move_text):
        """Counts half-moves (plies) in move text, ignoring comments/variations."""
        move_text = self.comment_pattern.sub("", move_text)
        move_text = self.variation_pattern.sub("", move_text)
        tokens = move_text.split()
        return len([t for t in tokens if not self.move_number_pattern.match(t)])


def is_rapid_game(game_lines):
    """Checks if the game is specifically a Rapid game (excludes Blitz)."""
    for line in game_lines:
        if line.strip().startswith("[Event "):
            line_lower = line.lower()
            # Must contain "rapid" and not contain "blitz"
            return "rapid" in line_lower and "blitz" not in line_lower
    return False


def filter_rapid_games(input_file, output_file, max_games=1_000_000, min_moves=15):
    """
    Filters Rapid games (excluding Blitz) with >15 moves, up to max_games limit.
    """
    move_counter = MoveCounter()
    written_games = 0
    current_game = []
    total_rapid = 0

    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:

        for line in infile:
            stripped = line.strip()

            if stripped.startswith("[Event "):
                if current_game:
                    # Process previous game
                    if is_rapid_game(current_game):
                        total_rapid += 1
                        # Extract moves and count
                        move_text = " ".join(
                            [l for l in current_game if not l.strip().startswith("[")]
                        )
                        plies = move_counter.count_plies(move_text)

                        if plies > min_moves * 2 and written_games < max_games:
                            outfile.write("\n".join(current_game) + "\n\n")
                            written_games += 1

                    current_game = []
                    if written_games >= max_games:
                        break

                current_game.append(line.rstrip("\n"))
            else:
                current_game.append(line.rstrip("\n"))

        # Process final game
        if current_game and written_games < max_games:
            if is_rapid_game(current_game):
                move_text = " ".join(
                    [l for l in current_game if not l.strip().startswith("[")]
                )
                plies = move_counter.count_plies(move_text)
                if plies > min_moves * 2:
                    outfile.write("\n".join(current_game) + "\n\n")
                    written_games += 1

    print(f"✅ Wrote {written_games} Rapid games (>15 moves) to {output_file}")
    print(f"Total Rapid games processed: {total_rapid}")


# Usage
input_pgn = "data/rapid_blitz_games.pgn"
output_pgn = "data/final_rapid_games.pgn"

filter_rapid_games(input_pgn, output_pgn)
