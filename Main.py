import pandas as pd

# Open the dataset up
chess_dataset = pd.read_csv("Training Data/games.csv")
# Create a mask
mask = chess_dataset["rules"] == "chess"

# Apply the mask to remove the rows
chess_dataset = chess_dataset[mask]
# Drop all unnecessary columns
chess_dataset = chess_dataset.drop(columns = ["white_username", "black_username", "white_id", "black_id", "time_class", "time_control", "rated", "rules", "rated", "fen", "white_result", "black_result"])
# Clean out PGN column
chess_dataset['pgn'] = chess_dataset['pgn'].str.split('\n\n').str[-1] # removes unnecessary info from the pgn string
chess_dataset['pgn'] = chess_dataset['pgn'].str.replace('\{\[%clk.*?\]\}', '', regex=True) # removes timestamps after every game
chess_dataset['pgn'] = chess_dataset['pgn'].str.replace(r'\d+\.\.\.', '', regex=True) # cleans up the pgn strings and formats it in the correct way

# Add a column that is the average of the ratings of the 2 players
chess_dataset["average_rating"] = ((chess_dataset['white_rating'] + chess_dataset['black_rating']) // 2).astype(int)
chess_dataset = chess_dataset.drop(columns = ["white_rating", "black_rating"])

# Create our model that can predict the elo of chess game
