import numpy as np


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

    def action_to_move(self, action):
        """
        Convert an action to a move.
        :param action: The action to convert.
        :return: The move.
        """
        return action // self.board_size, action % self.board_size

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


def main():
    """
    example of using HexBoard class.
    """
    hex_game = HexBoard(11)
    while hex_game.check_outcome() == hex_game.ONGOING:
        print(hex_game)
        row = int(input("Enter the row (0-10): "))
        col = int(input("Enter the column (0-10): "))
        move = (row, col)
        hex_game.make_move(move)
    print(hex_game)
    if hex_game.check_outcome() == hex_game.BLACK:
        print("Black wins")
    elif hex_game.check_outcome() == hex_game.WHITE:
        print("White wins")
    else:
        print("Draw")


if __name__ == '__main__':
    print("Running example of HexBoard class.")
    main()
