from queue import Queue
from threading import Thread
import LOGIC.Hex as Hex
import UI.GameScreen as GameScreen
import LOGIC.Puct as Puct
import LOGIC.NeuralNetwork as NN
import time

board_size = 5
neural_network = NN.NeuralNetwork(board_size)
neural_network.fetch_model()
puct = Puct.PuctMCTS(Hex.HexBoard(board_size), iterations=150, neural_network=neural_network)
q = Queue()
hex_game = Hex.HexBoard(board_size)
game_screen = GameScreen.Screen(hex_game.board, 20, "red", "blue", q)
t1 = Thread(target=game_screen.run)
t1.start()

while hex_game.check_outcome() == hex_game.ONGOING:
    if hex_game.current_player == Hex.HexBoard.BLACK:
        puct.board = hex_game
        move = puct.choose_move()
    else:
        move = q.get(True, None)
    hex_game.make_move(move)
    time.sleep(0.02)  # sleep for visual effect
time.sleep(0.1)
if hex_game.check_outcome() == hex_game.BLACK:
    game_screen.winner = "RED"
    print("Red wins")
else:
    game_screen.winner = "BLUE"
    print("Blue wins")
game_screen.game_ends = True

t1.join()
