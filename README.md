# GuessTheEloAI
### Used Libraries
####
* Pandas
* Tensorflow
* Sklearn
* Regex
* Chess
* Stockfish

### How it works
#### The dataset that was used to train the model consists of just the average rating of the game and the PGN for the game. The PGN string had to be converted to a numerical format so that the model could be trained. So, in the `pgn_process.py` file, the PGN for all games were preprocessed by splitting the moves into the amount of good, bad, book, forced moves that were played in the game. It also stored the opening that was played in the game. The opening and the amount of book moves played were found according to the `opening_book/eco.pgn` file. The AI was then trained using this preprocessed data.

<br>  
<br>  

The AI in its current state isn't that great. It can only predict the elo correctly around 20% of the time on average, however, it is usually only off by 100 to 200 elo points off. This is mainly because of how the data was preprocessed. Because the PGN was only split up into good and bad moves, there wasn't a lot to go off of. Also, I made it so that every move that lowers the stockfish eval by 2 was bad move, which is also a vague definition of a bad move. Over time, I'll try to make this AI stronger by preprocessing the data better.
