import re

# Define the path to the PGN file
pgn_file = "opening_book/eco.pgn"

# Initialize the dictionary to store the openings
openings = {}

# Open the PGN file
with open(pgn_file, "r") as file:
    pgn_text = file.read()

# Extract the opening data using regular expressions
pattern = r'\[ECO "(.*?)"]\n\[Opening "(.*?)"](?:\n\[Variation "(.*?)"])?\n\n(.*?)\n\n'
matches = re.findall(pattern, pgn_text, re.DOTALL)


# Iterate over the matches and store the data in the dictionary
for match in matches:
    eco = match[0]
    opening = match[1]
    variation = match[2] if match[2] else None
    moves = match[3].strip().split("\n")
    move_str = ""

    for move in moves:
        move = move.strip()
        if move.startswith("1."):
            move_str = move[:-1]  # Remove the asterisk at the end
        else:
            move_str += f" {move}"

    if variation:
        # opening_key = f"{eco} - {opening} - {variation}"
        opening_key = [eco, opening, variation]
    else:
        # opening_key = f"{eco} - {opening}"
        opening_key = [eco, opening]

    if move_str not in openings:
        openings[move_str] = opening_key

openings2 = {}

for key, value in openings.items():
    new_key = key.rstrip(" *")
    openings2[new_key] = value

openings = openings2
del openings2

