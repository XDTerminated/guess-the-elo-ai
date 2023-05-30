import chess.pgn
import io
import chess
import chess.engine
import pandas as pd


def get_piece_count_fen(fen):

    board = chess.Board()

    board.set_fen(fen)

    piece_mapping = {
    chess.KING: 0,
    chess.QUEEN: 1,
    chess.ROOK: 2,
    chess.BISHOP: 3,
    chess.KNIGHT: 4,
    chess.PAWN: 5
    }

    piece_count = [[0] * 6, [0] * 6]

    for piece in board.piece_map().values():
        color = piece.color
        piece_type = piece.piece_type
        piece_index = piece_mapping[piece_type]
        piece_count[color][piece_index] += 1

    return piece_count[chess.WHITE], piece_count[chess.BLACK]

def get_piece_count(pgn):
    piece_count = []

    board = chess.Board()
    game = chess.pgn.read_game(io.StringIO(pgn))

    node = game
    while node.variations:
        white, black = get_piece_count_fen(board.fen())
        piece_count.append([white, black])
        board.push(node.variations[0].move)
        node = node.variations[0]

    return piece_count

def get_stockfish_eval(pgn):
    eval_list = []

    board = chess.Board()
    game = chess.pgn.read_game(io.StringIO(pgn))

    node = game
    while node.variations:
        eval = get_eval(board)
        eval_list.append(eval)
        board.push(node.variations[0].move)
        node = node.variations[0]

    return eval_list

def get_eval(board):
    stockfish = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/Cellar/stockfish/15.1/bin/stockfish")
    result = stockfish.analyse(board, chess.engine.Limit(time=0.005))
    stockfish.quit()
    return result["score"].relative.cp/100


chess_data = pd.read_csv("data/games.csv")
print(get_stockfish_eval(chess_data["pgn"].iloc[0]))

