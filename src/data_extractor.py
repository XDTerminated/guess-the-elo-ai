import zstandard as zstd
import io


def is_rapid_or_blitz(game_buffer):
    """Checks if the game header indicates a rapid or blitz game."""
    for line in game_buffer:
        if line.startswith("[Event "):
            if "blitz" in line.lower() or "rapid" in line.lower():
                return True
    return False


def extract_rapid_and_blitz_games(input_file, output_file):
    """Extracts all games (including moves) that are rapid or blitz from a compressed PGN file."""
    game_count = 0
    rapid_blitz_count = 0
    game_buffer = []

    with open(input_file, "rb") as compressed, open(
        output_file, "w", encoding="utf-8"
    ) as output:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
            for line in text_stream:
                stripped_line = line.strip()
                if stripped_line.startswith("[Event "):
                    if game_buffer:
                        if is_rapid_or_blitz(game_buffer):
                            output.write("\n".join(game_buffer) + "\n\n")
                            rapid_blitz_count += 1
                        game_count += 1
                        game_buffer = []
                game_buffer.append(line.rstrip("\n"))

            if game_buffer:
                if is_rapid_or_blitz(game_buffer):
                    output.write("\n".join(game_buffer) + "\n\n")
                    rapid_blitz_count += 1
                game_count += 1

    print(
        f"Extracted {rapid_blitz_count} rapid and blitz games (with moves) out of {game_count} games to {output_file}"
    )


input_filename = "data/lichess_db_standard_rated_2025-01.pgn.zst"
output_filename = "data/rapid_blitz_games.pgn"

extract_rapid_and_blitz_games(input_filename, output_filename)
