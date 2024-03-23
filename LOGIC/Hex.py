import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

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
        if row > 0 and col < self.board_size - 1:
            if self.board[row - 1][col + 1] == player:
                neighbors.append((row - 1, col + 1))
        if row < self.board_size - 1 and col > 0:
            if self.board[row + 1][col - 1] == player:
                neighbors.append((row + 1, col - 1))
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


class HexNet(nn.Module):
    def __init__(self, board_size):
        super(HexNet, self).__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * board_size * board_size, 128)
        self.fc_value = nn.Linear(128, 1)
        self.fc_policy = nn.Linear(128, board_size * board_size)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))
        x = x.view(-1, 128 * self.board_size * self.board_size)
        x = f.relu(self.fc1(x))
        value = torch.tanh(self.fc_value(x))
        policy = f.softmax(self.fc_policy(x), dim=1)
        return value, policy


class PUCT:
    def __init__(self, network, c=1.0):
        self.network = network
        self.c = c

    def search(self, game_state):
        # PUCT search algorithm

        # Get the current player
        player = game_state.current_player

        # Get the legal moves
        legal_moves = game_state.legal_moves()

        # Initialize the action values
        action_values = torch.zeros(len(legal_moves))

        # Get the current board state
        board_state = torch.tensor(game_state.board, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Get the value and policy from the network
        value, policy = self.network(board_state)

        # Get the value and policy from the network
        value = value.item()
        policy = policy.squeeze().detach().numpy()

        # Calculate the action values
        for i, move in enumerate(legal_moves):
            action_values[i] = (1 - self.c) * policy[move[0] * game_state.board_size + move[1]] + self.c * value

        # Get the best action
        best_action = legal_moves[action_values.argmax()]

        # Return the best action
        return best_action

    def update(self, game_state, action):
        # Update the game state with the selected action
        game_state.make_move(action)

        # Return the updated game state
        return game_state


def train_network(board_size=11):
    # Define hyperparameters
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 64

    # Initialize game, network, optimizer
    hex_game = HexBoard()  # Pass board_size here
    hex_network = HexNet(board_size=board_size)  # Pass board_size to HexNet
    puct = PUCT(network=hex_network)
    optimizer = torch.optim.Adam(hex_network.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        # Generate training data through self-play
        training_data = []
        for _ in range(batch_size):
            game_state = hex_game.clone()
            while game_state.check_outcome() == HexBoard.ONGOING:
                action = puct.search(game_state)
                game_state = puct.update(game_state, action)
            training_data.append(game_state)

        # Train the network
        for game_state in training_data:
            # Get the current player
            player = game_state.current_player

            # Get the legal moves
            legal_moves = game_state.legal_moves()

            # Get the current board state
            board_state = torch.tensor(game_state.board, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            # Get the value and policy from the network
            value, policy = hex_network(board_state)

            # Calculate the target value and policy
            target_value = torch.tensor(game_state.check_outcome(), dtype=torch.float32)
            target_policy = torch.zeros(board_size * board_size)  # Use board_size here
            for move in legal_moves:
                target_policy[move[0] * board_size + move[1]] = 1  # Use board_size here
            target_policy = target_policy / target_policy.sum()

            # Calculate the loss
            loss = f.mse_loss(value, target_value) + f.kl_div(policy.log(), target_policy)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print the loss
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the trained network
    torch.save(hex_network.state_dict(), 'hex_network.pth')

    # Return the trained network
    return hex_network


def test_model():
    # Load trained model
    hex_network = HexNet(board_size=11)
    hex_network.load_state_dict(torch.load('hex_network.pth'))
    hex_network.eval()

    # Test against human players or MCTS
    hex_game = HexBoard(board_size=11)
    puct = PUCT(network=hex_network)
    while hex_game.check_outcome() == HexBoard.ONGOING:
        print(hex_game)
        if hex_game.current_player == HexBoard.BLACK:
            action = puct.search(hex_game)
            hex_game.make_move(action)
        else:
            row, col = map(int, input('Enter row and column: ').split())
            hex_game.make_move((row, col))
    print(hex_game)
    outcome = hex_game.check_outcome()
    if outcome == HexBoard.WHITE:
        print('White won!')
    elif outcome == HexBoard.BLACK:
        print('Black won!')


def main():
    """
    Example usage of the Hex board.
    2 players play against each other.
    """
    hex_game = HexBoard(board_size=11)
    while hex_game.check_outcome() == HexBoard.ONGOING:
        print(hex_game)
        row, col = map(int, input('Enter row and column: ').split())
        hex_game.make_move((row, col))
    print(hex_game)
    outcome = hex_game.check_outcome()
    if outcome == HexBoard.WHITE:
        print('White won!')
    elif outcome == HexBoard.BLACK:
        print('Black won!')


if __name__ == '__main__':
    main()
