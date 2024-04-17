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
    def __init__(self, board, move=None, parent=None, prior=0):
        self.visits = 0
        self.wins = 0
        self.board = board
        self.parent = parent
        self.move = move
        self.prior = prior
        self.q = 0
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
        Select a child node using the Puct (predictor + UCT) formula:
        PUCT(a) = q + c * p * sqrt(N) / (1 + n)
        where:
        q = exploitation term (average reward of the child node)
        p = prior probability of selecting the child node
        N = total number of visits of the parent node
        n = number of visits of the child node
        c = exploration parameter
        """
        assert self.visits > 0, "Parent has not been visited"
        c = 4
        best_score = float('-inf')
        best_child = None
        for child in self.children:
            exploitation = child.wins / child.visits if child.visits > 0 else 0
            exploration = self.prior * math.sqrt(self.visits) / (1 + child.visits)
            puct_score = exploitation + c * exploration
            if puct_score > best_score:
                best_score = puct_score
                best_child = child
        return best_child

    def expand(self, pi_array):
        """
        Expand the node by adding a new child node.
        """
        pi_array = pi_array.flatten()
        for action, prior in enumerate(pi_array):
            print(prior)
            if prior > 0:
                next_board = self.board.clone()
                col, row = action % self.board.board_size, action // self.board.board_size
                next_board.make_move((row, col))
                child_node = Puct_Node(next_board, action, self, prior)
                self.children.append(child_node)
        self.untried_actions = []

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
        self.q += (result - self.q) / self.visits
        if self.parent:
            self.parent.backpropagation(result)


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
            # Selection
            while not node.is_terminal() and not node.untried_actions:
                node = node.select_child()
            # Expansion
            if not node.is_terminal():
                pi, v = self.neural_network.model.predict(
                    node.board.get_encoded_board().reshape(1, 3, self.board.board_size, self.board.board_size)
                )
                valid_moves = node.board.legal_moves()
                mask = np.zeros([self.board.board_size] * 2)
                for col, row in valid_moves:
                    mask[row][col] = 1
                mask = mask.flatten()
                pi = pi * mask
                pi = pi / np.sum(pi)
                val = v[0][0]
                node.expand(pi)
            # Backpropagation
            node.backpropagation(val)
        # Choose the best move based on the number of visits
        best_child = max(root.children, key=lambda x: x.visits)
        return best_child.move

    def choose_move_greedy(self):
        encoded_board = self.neural_network.get_encoded_board(self.board.board)
        v, pi = self.neural_network.model.predict(
            encoded_board.reshape(1, 3, self.board.board_size, self.board.board_size)
        )
        pi = self.neural_network.get_decoded_policy(pi, self.board)
        print(pi)
        print(self.board)
        best_move = np.argmax(pi)
        print(best_move)

        return best_move % self.board.board_size, best_move // self.board.board_size


def main():
    """
    simple example of using MCTS for Hex.
    """
    board_size = 9
    neural_network = NN.NeuralNetwork(board_size)
    puct = PuctMCTS(Hex.HexBoard(board_size), iterations=1000, neural_network=neural_network)
    neural_network.fetch_weights()
    print(neural_network.model.weights)
    q = Queue()
    hex_game = Hex.HexBoard(9)
    game_screen = GameScreen.Screen(hex_game.board, 20, "red", "blue", q)
    t1 = Thread(target=game_screen.run)
    t1.start()

    while hex_game.check_outcome() == hex_game.ONGOING:
        if hex_game.current_player == Hex.HexBoard.BLACK:
            puct.board = hex_game
            move = puct.choose_move_greedy()
        else:
            move = q.get(True, None)
        hex_game.make_move(move)
        time.sleep(0.2)  # sleep for visual effect
    time.sleep(1)
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
