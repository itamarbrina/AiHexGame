from queue import Queue
from threading import Thread
import LOGIC.Hex as Hex
import UI.GameScreen as GameScreen


q = Queue()
hex_game = Hex.HexBoard(11)
game_screen = GameScreen.Screen(hex_game.board, 20, "red", "blue", q)
t1 = Thread(target=game_screen.run)
t1.start()

while hex_game.check_outcome() == hex_game.ONGOING:
    move = q.get(True, None)
    hex_game.make_move(move)
if hex_game.check_outcome() == hex_game.BLACK:
    print("Black wins")
elif hex_game.check_outcome() == hex_game.WHITE:
    print("White wins")

t1.join()
