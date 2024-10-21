import json
from clean_pgn import clean_pgn


with open("data/external/openings.json", "r") as file:
    openings = json.load(file)

file.close()

openings_dict = {}

for i in range(len(openings)):
    opening_moves = openings[i]["moves"]
    opening_eco = openings[i]["eco"]

    openings_dict[clean_pgn(opening_moves)] = opening_eco

with open("data/external/openings_condensed.json", "w") as file:
    json.dump(openings_dict, file)
