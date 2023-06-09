from stockfish import Stockfish
import chess
import chess.pgn
import io
import chess.polyglot
import pandas as pd

stockfish = Stockfish(path = "/opt/homebrew/Cellar/stockfish/15.1/bin/stockfish", depth = 10, parameters = {"Threads": 2, "Hash": 2048, "Minimum Thinking Time": 1})
opening_book_path = "opening_book/komodo.bin"

board = chess.Board()
board.push_san("e4")
board.push_san("e5")
board.push_san("Nf3")
board.push_san("Nc6")


with chess.polyglot.open_reader(opening_book_path) as reader:
    for entry in reader.find_all(board):
        print(entry.move, entry.weight, entry.learn)



def get_game_report(pgn):
    board = chess.Board()
    game = chess.pgn.read_game(io.StringIO(pgn))

    openning = 0
    brilliant = 0
    great = 0
    best = 0
    excellent = 0
    good = 0
    missed_win = 0
    inaccurate = 0
    mistake = 0
    blunder = 0

    node = game
    while node.variations:
        fen = board.fen()
        stockfish.set_fen_position(fen)
        top_moves = stockfish.get_top_moves(7)
        move_made = node.variations[0].move

        board.push(move_made)
        node = node.variations[0]