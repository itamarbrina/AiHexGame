import math
import random
import time
from queue import Queue
from threading import Thread
import numpy as np
import LOGIC.NeuralNetwork as NN

import LOGIC.Hex as Hex
import LOGIC.Mcts as Mcts
import UI.GameScreen as GameScreen

"""
Monte Carlo Tree Search
"""


class Puct_Node:
    def __init__(self, board, move=None, parent=None, prior=0, wins_avg=0):
        self.visits = 0
        self.wins_avg = wins_avg
        self.board = board
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children = []

    def is_terminal(self):
        """
        Check if the node is a terminal node.
        """
        return self.board.check_outcome() != Hex.HexBoard.ONGOING

    def select_child(self):
        """
        Select a child node using the Puct (predictor + UCT) formula:
        PUCT(a) = wins_avg + c * p * sqrt(N) / (1 + n)
        where:
        q = exploitation term (average reward of the child node)
        p = prior probability of selecting the child node
        N = total number of visits of the parent node
        n = number of visits of the child node
        c = exploration parameter
        """
        assert self.visits > 0, "Parent has not been visited"
        c = math.sqrt(2)
        best_score = float('-inf')
        best_child = None
        for child in self.children:
            puct_score = child.wins_avg + c * child.prior * math.sqrt(self.visits) / (1 + child.visits)
            if puct_score > best_score:
                best_score = puct_score
                best_child = child
        return best_child

    def expand(self, pi):
        """
        Expand the node by adding a new child node.
        """
        pi = pi.flatten()
        for action, prob in enumerate(pi):
            if prob > 0:
                next_board = self.board.clone()
                row = action // self.board.board_size
                col = action % self.board.board_size
                next_board.make_move((col, row))
                child_node = Puct_Node(next_board, move=(col, row), parent=self, prior=prob)
                self.children.append(child_node)

    # def simulate(self):
    #     """
    #     Simulate a random game from the current node.
    #     """
    #     board = self.board.clone()
    #     while board.check_outcome() == Hex.HexBoard.ONGOING:
    #         move = board.legal_moves()[np.random.randint(0, len(board.legal_moves()))]
    #         board.make_move(move)
    #     return board.check_outcome()

    def backpropagation(self, value):
        """
        Updates the node's statistics after a simulation.
        """
        self.visits += 1
        if self.board.current_player == Hex.HexBoard.WHITE:
            value = 1 - value

        self.wins_avg += (value - self.wins_avg) / self.visits

        if self.parent:
            self.parent.backpropagation(value)


class PuctMCTS:
    """
    Runs the Monte Carlo Tree Search algorithm.
    """

    def __init__(self, board, iterations=1000, neural_network=None):
        self.board = board
        self.iterations = iterations
        self.neural_network = neural_network

    def choose_move(self):
        """
        Choose the best move using the PUCT algorithm.
        """
        root = Puct_Node(self.board)
        for _ in range(self.iterations):
            node = root
            value = node.board.check_outcome()
            if value == node.board.current_player:
                value = 1
            elif value == node.board.other_player(node.board.current_player):
                value = 0
            else:
                value = 0.5

            # Selection
            while not node.is_terminal() and node.children != []:
                node = node.select_child()
            # Expansion
            if not node.is_terminal() and node.children == []:
                value, pi = self.neural_network.model.predict(
                    self.neural_network.get_encoded_board(node.board.board)
                    .reshape(1, 3, self.board.board_size, self.board.board_size)
                )
                pi = self.neural_network.get_decoded_policy(pi, node.board)
                node.expand(pi)
                value = value[0][0]
            # Backpropagation
            node.backpropagation(value)
        # Choose the best move based on the number of visits
        best_child = max(root.children, key=lambda x: x.visits)
        return best_child.move

    def choose_move_greedy(self):
        encoded_board = self.neural_network.get_encoded_board(self.board.board)
        v, pi = self.neural_network.model.predict(
            encoded_board.reshape(1, 3, self.board.board_size, self.board.board_size)
        )
        pi = self.neural_network.get_decoded_policy(pi, self.board)
        print(v)
        print(pi)
        print(self.board)
        best_move = np.argmax(pi)
        print(best_move)

        return best_move % self.board.board_size, best_move // self.board.board_size


def main():
    """
    simple example of using MCTS for Hex.
    """
    board_size = 5
    neural_network = NN.NeuralNetwork(board_size)
    neural_network.fetch_model()
    puct = PuctMCTS(Hex.HexBoard(board_size), iterations=150, neural_network=neural_network)
    q = Queue()
    hex_game = Hex.HexBoard(board_size)
    game_screen = GameScreen.Screen(hex_game.board, 20, "red", "blue", q)
    t1 = Thread(target=game_screen.run)
    t1.start()

    while hex_game.check_outcome() == hex_game.ONGOING:
        if hex_game.current_player == Hex.HexBoard.BLACK:
            puct.board = hex_game
            # move = puct.choose_move_greedy()
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


if __name__ == "__main__":
    print("Running example of MCTS for Hex.")
    main()
