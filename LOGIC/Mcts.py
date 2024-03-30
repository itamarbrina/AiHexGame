import LOGIC.Hex as Hex
import random
import math
import numpy as np
from queue import Queue
from threading import Thread
import UI.GameScreen as GameScreen

"""
Monte Carlo Tree Search
"""


class Node:
    def __init__(self, board, move=None, parent=None):
        self.visits = 0
        self.wins = 0
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.untried_actions = board.legal_moves()
        random.shuffle(self.untried_actions)

    def is_terminal(self):
        """
        Check if the node is a terminal node.
        """
        return self.board.check_outcome() != Hex.HexBoard.ONGOING

    def select_child(self):
        """
        Select a child node using the UCT (Upper Confidence Bound for Trees) formula.
        """
        assert self.visits > 0, "Parent has not been visited"
        c = math.sqrt(2)
        # Calculate UCT score for each child and select the child with the highest score
        best_score = float('-inf')
        best_child = None
        for child in self.children:
            # Calculate the exploitation and exploration terms separately for clarity
            exploitation = child.wins / child.visits if child.visits > 0 else 0
            exploration = math.sqrt(math.log(self.visits) / (1 + child.visits))  # +1 to ensure non-zero division
            uct_score = exploitation + c * exploration
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        return best_child

    def expand(self):
        """
        Expand the node by adding a new child node.
        """
        action = self.untried_actions.pop()
        next_board = self.board.clone()
        next_board.make_move(action)
        child_node = Node(next_board, action, self)
        self.children.append(child_node)
        return child_node

    def simulate(self):
        """
        Simulate a random game from the current node.
        """
        board = self.board.clone()
        while board.check_outcome() == Hex.HexBoard.ONGOING:
            move = board.legal_moves()[np.random.randint(0, len(board.legal_moves()))]
            board.make_move(move)
        return board.check_outcome()

    def backpropagation(self, result):
        """
        Updates the node's statistics after a simulation.
        """
        self.visits += 1
        if self.board.other_player(self.board.current_player) == result:
            self.wins += 1
        if result == Hex.HexBoard.DRAW:
            self.wins += 0.5
        if self.parent:
            self.parent.backpropagation(result)


class MCTS:
    """
    Runs the Monte Carlo Tree Search algorithm.
    """
    def __init__(self, board, iterations=1000):
        self.board = board
        self.iterations = iterations

    def choose_move(self):
        """
        Choose the best move using the MCTS algorithm.
        """
        root = Node(self.board)
        for _ in range(self.iterations):
            node = root
            # Select
            while not node.untried_actions and not node.is_terminal():
                node = node.select_child()
            # Expand
            if node.untried_actions and not node.is_terminal():
                node = node.expand()
            # Simulate
            result = node.simulate()
            # Backpropagation
            node.backpropagation(result)
        return max(root.children, key=lambda x: x.visits).move


def main():
    """
    simple example of using MCTS for Hex.
    """
    q = Queue()
    mcts = MCTS(Hex.HexBoard(6), iterations=2500)
    hex_game = Hex.HexBoard(6)
    game_screen = GameScreen.Screen(hex_game.board, 20, "red", "blue", q)
    t1 = Thread(target=game_screen.run)
    t1.start()

    while hex_game.check_outcome() == hex_game.ONGOING:
        if hex_game.current_player == Hex.HexBoard.BLACK:
            mcts.board = hex_game
            move = mcts.choose_move()
        else:
            move = q.get(True, None)
        hex_game.make_move(move)
    if hex_game.check_outcome() == hex_game.BLACK:
        print("Black wins")
    elif hex_game.check_outcome() == hex_game.WHITE:
        print("White wins")

    t1.join()


if __name__ == "__main__":
    print("Running example of MCTS for Hex.")
    main()
