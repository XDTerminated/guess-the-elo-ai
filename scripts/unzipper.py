# Source: https://github.com/ManasiTilak/pgn_zst_unzipper

import zstandard
import shutil
from tqdm import tqdm


def decompress_zstd(input_file, output_file):
    with open(input_file, "rb") as compressed_file:
        decompressor = zstandard.ZstdDecompressor()
        with decompressor.stream_reader(compressed_file) as reader:
            with open(output_file, "wb") as decompressed_file:
                decompressed_file.write(reader.read())


def main():
    input_file = "data/raw/lichess_game_2015.pgn.zst"
    output_file = "data/raw/lichess_game_2015_unzipped.pgn"

    decompress_zstd(input_file, output_file)
    print("Decompression completed.")


if __name__ == "__main__":
    main()
