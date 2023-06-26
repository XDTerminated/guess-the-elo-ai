from stockfish import Stockfish
import chess.engine
import chess
import chess.pgn
import io
import pandas as pd
from extract_openings import openings

df = pd.read_csv("games.csv")


stockfish = Stockfish(path = "/opt/homebrew/Cellar/stockfish/15.1/bin/stockfish", depth = 8, parameters = {"Threads": 1, "Hash": 2048, "Minimum Thinking Time": 1})



def get_game_report(pgn):
    board = chess.Board()
    board2 = chess.Board()
    game = chess.pgn.read_game(io.StringIO(pgn))

    opening = True

    book = 0
    opening_name = ""
    best = 0
    bad = 0
    forced = 0
    length = 0

    moves = []

    node = game
    while node.variations:
        fen = board2.fen()
        move_made = node.variations[0].move
        moves.append(move_made)

        board2.push(move_made)
        s = board.variation_san(moves)

        length+=1
        node = node.variations[0]
        if opening:
            try:
                a = openings[s]
                book+=1
                opening_name = a[1]
                continue

            except:
                is_opening = False
                for key, value in openings.items():
                    if s in key:
                        book+=1
                        opening_name = value[1]
                        is_opening = True
                        break
                
                if not is_opening:
                    opening = False
                else:
                    continue
            
        c = stockfish.get_evaluation()["value"]
        stockfish.set_fen_position(fen)
        d = stockfish.get_evaluation()["value"]
        top_moves = stockfish.get_top_moves(7)



        if (len(top_moves) == 1):
            forced+=1

        else:
            if str(move_made) == top_moves[0]["Move"]:
                best+=1
                continue
            
            if board2.turn:
                if d - c <= -2:
                    bad+=1

            if not board2.turn:
                if d- c >= 2:
                    bad+=1

    return book, best, forced, opening_name, bad, length

def process_pgn():
    book = []
    best = []
    forced = []
    opening_name = []
    bad = []
    length = []


    for i in range(11875):
        print(i)
        b1, b2, f, o, b3, l = get_game_report(df["pgn"].iloc[i])
        book.append(b1)
        best.append(b2)
        forced.append(f)
        opening_name.append(o)
        bad.append(b3)
        length.append(l)

    df["book"] = book
    df["best"] = best
    df["forced"] = forced
    df["opening"] = opening_name
    df["bad"] = bad
    df["length"] = length

    df.to_csv("games.csv")
