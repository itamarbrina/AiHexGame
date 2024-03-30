# import keras
import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as f

# noinspection GrazieInspection
class HexBoard:
    """
    This class is a representation of the Hex board. It has methods to make moves, check for a winner,
    and clone the board.
    """

    EMPTY = 0
    BLACK = 1
    WHITE = -1
    ONGOING = -17
    DRAW = 0

    def __init__(self, board_size=11, queue=None):
        """
        Initialize the Hex board.
        :param board_size: Size of the board (default is 11 x 11).
        """
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = self.BLACK
        self.game_history = []

    def other_player(self, player):
        """
        Switch between players, for inner use.
        :param player: The current player.
        :return:  The other player.
        """
        return self.WHITE if player == self.BLACK else self.BLACK

    def clone(self):
        """
        Create a deep copy of the board, with the same position and history.
        :return: A cloned instance of the board.
        """
        copy = HexBoard(self.board_size)
        copy.current_player = self.current_player
        copy.board = np.copy(self.board)
        copy.game_history = self.game_history[:]
        return copy

    def last_move(self):
        """
        Get the last move made in the game.
        :return: Last move coordinates (row, col).
        """
        if len(self.game_history) == 0:
            return None
        return self.game_history[-1]

    def make_move(self, move):
        """
        Make a move on the board if it is legal.
        :param move: Move coordinates (row, col).
        """
        row = move[0]
        col = move[1]
        if 0 <= row < self.board_size and 0 <= col < self.board_size and self.board[row][col] == 0:
            self.board[row][col] = self.current_player
            self.game_history.append(move)
            self.current_player = self.other_player(self.current_player)

    def unmake_move(self):
        """
        Undo the last move made in the game.
        """
        if len(self.game_history) > 0:
            move = self.game_history.pop()
            self.board[move[0]][move[1]] = 0
            self.current_player = self.other_player(self.current_player)

    def legal_moves(self):
        """
        Get all legal moves available on the board.
        :return: List of legal move coordinates (row, col).
        """
        return [(i, j) for i in range(self.board_size) for j in range(self.board_size) if self.board[i][j] == 0]

    def get_neighbors(self, player, move):
        """
        Get all the same color neighbors of a move to find winning path, for inner use.
        :param player: The current player.
        :param move: Move coordinates (row, col).
        :return: List of neighbors coordinates (row, col).
        """
        row = move[0]
        col = move[1]
        neighbors = []
        if row > 0:
            if self.board[row - 1][col] == player:
                neighbors.append((row - 1, col))
        if row < self.board_size - 1:
            if self.board[row + 1][col] == player:
                neighbors.append((row + 1, col))
        if col > 0:
            if self.board[row][col - 1] == player:
                neighbors.append((row, col - 1))
        if col < self.board_size - 1:
            if self.board[row][col + 1] == player:
                neighbors.append((row, col + 1))
        # if row > 0 and col < self.board_size - 1:
        #     if self.board[row - 1][col + 1] == player:
        #         neighbors.append((row - 1, col + 1))
        # if row < self.board_size - 1 and col > 0:
        #     if self.board[row + 1][col - 1] == player:
        #         neighbors.append((row + 1, col - 1))
        if row > 0 and col > 0:
            if self.board[row - 1][col - 1] == player:
                neighbors.append((row - 1, col - 1))
        if row < self.board_size - 1 and col < self.board_size - 1:
            if self.board[row + 1][col + 1] == player:
                neighbors.append((row + 1, col + 1))
        return neighbors

    def search_for_win_path(self, player, move):
        """
        Search for a winning path from one side of the board to the other, for inner use.
        This function implements some kind of DFS to find a path from one side to the other.
        :param player: The player to check for a win path (makes the last move).
        :param move: Move coordinates (row, col).
        :return: True if a win path is found, False otherwise.
        """
        visited = [[False for _ in range(self.board_size)] for _ in range(self.board_size)]
        visited[move[0]][move[1]] = True
        neighbors = self.get_neighbors(player, move)
        while neighbors:
            neighbor = neighbors.pop()
            if visited[neighbor[0]][neighbor[1]]:
                continue
            visited[neighbor[0]][neighbor[1]] = True
            neighbors += self.get_neighbors(player, neighbor)
        if player == self.BLACK:
            if any(visited[0][col] for col in range(self.board_size)) \
                    and any(visited[self.board_size - 1][col] for col in range(self.board_size)):
                return True
        if player == self.WHITE:
            if any(visited[row][0] for row in range(self.board_size)) \
                    and any(visited[row][self.board_size - 1] for row in range(self.board_size)):
                return True
        return False

    def check_outcome(self):
        """
        Determine the outcome of the game.
        :return: Outcome of the game (BLACK, WHITE, or ONGOING).
        """
        if len(self.game_history) == 0:
            return self.ONGOING

        if len(self.game_history) == self.board_size ** 2:
            return self.DRAW

        last_move = self.game_history[-1]
        player = self.board[last_move[0]][last_move[1]]
        if self.search_for_win_path(player, last_move):
            return player

        return self.ONGOING

    def __str__(self):
        """
        String representation of the game board in the console.
        This is use for example and testing purposes.
        :return: The string representation of the game board.
        """
        symbols = {self.EMPTY: '.', self.BLACK: 'X', self.WHITE: 'O'}
        board_str = ''
        for i in range(self.board_size):
            board_str += ' ' * (self.board_size - i - 1)
            for j in range(self.board_size):
                board_str += symbols[self.board[i][j]] + ' '
            board_str += '\n'
        return board_str


# class NeuralNetwork:
#     def __init__(self, input_shape):
#         self.model = self.build_model(input_shape)
#
#     def build_model(self, input_shape):
#         model = keras.Sequential([
#             keras.layers.Input(input_shape),
#             keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#             keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#             keras.layers.Flatten(),
#             keras.layers.Dense(128, activation='relu'),
#             keras.layers.Dense(1, activation='tanh')
#         ])
#         model.compile(optimizer='adam', loss='mse')
#         return model
#
#     def predict(self, state):
#         return self.model.predict(np.expand_dims(state, axis=0))[0][0]
#
#
# class MCTSNode:
#     def __init__(self, state):
#         self.state = state
#         self.children = []
#         self.visits = 0
#         self.value = 0
#
#     def expand(self):
#         legal_moves = self.state.legal_moves()
#         for move in legal_moves:
#             new_state = self.state.clone()
#             new_state.make_move(move)
#             child_node = MCTSNode(new_state)
#             self.children.append(child_node)
#
#     def select_child(self, exploration_constant=1.0):
#         values = [child.value + exploration_constant * np.sqrt(np.log(self.visits) / (child.visits + 1)) for child in
#                   self.children]
#         return self.children[np.argmax(values)]
#
#     def update(self, value):
#         self.visits += 1
#         self.value += value
#
#
# class MCTS:
#     def __init__(self, root_state, neural_network):
#         self.root = MCTSNode(root_state)
#         self.neural_network = neural_network
#
#     def search(self, num_simulations=100):
#         for _ in range(num_simulations):
#             node = self.root
#             while node.children:
#                 node = node.select_child()
#             if node.visits == 0:
#                 node.expand()  # Ensure that the node is expanded before selecting a child
#             if node.children:
#                 value = self.simulate(node.state)
#                 node.update(value)
#
#     def simulate(self, state):
#         _state = state.clone()
#         while _state.check_outcome() == _state.ONGOING:
#             legal_moves = _state.legal_moves()
#             move = legal_moves[np.random.randint(len(legal_moves))]
#             _state.make_move(move)
#         outcome = _state.check_outcome()
#         return outcome
#
#     def evaluate(self, state):
#         return self.neural_network.predict(state.board)
#
#     def get_best_move(self):
#         best_move = max(self.root.children, key=lambda x: x.visits)
#         return best_move.state.last_move()
#
#
# # Example usage:
# # Main function to simulate a game between a player and AI
# def main():
#     board = HexBoard(board_size=11)
#     nn = NeuralNetwork(input_shape=(board.board_size, board.board_size, 1))
#     mcts = MCTS(board, nn)
#
#     print("Welcome to Hex! You are playing as BLACK.")
#     print(board)  # Print initial board state
#
#     while board.check_outcome() == board.ONGOING:
#         if board.current_player == board.BLACK:
#             # Player's turn
#             while True:
#                 try:
#                     row = int(input("Enter the row (0-10): "))
#                     col = int(input("Enter the column (0-10): "))
#                     move = (row, col)
#                     if move in board.legal_moves():
#                         break
#                     else:
#                         print("Invalid move. Please try again.")
#                 except ValueError:
#                     print("Invalid input. Please enter integers.")
#
#             board.make_move(move)
#             print(board)  # Print updated board state
#
#         else:
#             # AI's turn
#             print("AI is thinking...")
#             mcts.search(num_simulations=500)
#             best_move = mcts.get_best_move()
#             board.make_move(best_move)
#             print("AI's move:", best_move)
#             print(board)  # Print updated board state
#
#     # Game outcome
#     outcome = board.check_outcome()
#     if outcome == board.BLACK:
#         print("Congratulations! You win!")
#     elif outcome == board.WHITE:
#         print("AI wins. Better luck next time!")
#     else:
#         print("It's a draw.")

# # Run the main function
# if __name__ == "__main__":
#     main()
