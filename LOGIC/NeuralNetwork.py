import keras
import numpy as np
import LOGIC.Puct as Puct
import LOGIC.Hex as Hex
import LOGIC.Mcts as Mcts


class NeuralNetwork:
    """
    Simple neural network for getting value and policy.
    """

    def __init__(self, board_size, learning_rate=0.01):
        self.board_size = board_size
        self.conv1 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        self.conv2 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.conv3 = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')
        self.flatten = keras.layers.Flatten()
        self.policy = keras.layers.Dense(board_size ** 2, activation='softmax')
        self.value = keras.layers.Dense(1, activation='tanh')
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        """
        build the neural network model that get state as input and return value and policy as output.
        :return: policy and value model.
        """
        input_layer = keras.layers.Input((3, self.board_size, self.board_size))
        x = self.conv1(input_layer)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        policy = self.policy(x)
        value = self.value(x)
        model = keras.Model(inputs=input_layer, outputs=[value, policy])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss=['mean_squared_error', 'categorical_crossentropy'])
        self.model = model
        return model

    def get_encoded_board(self, board):
        """
        Encode the board in a format that can be used by the neural network.
        :return: The encoded board.
        """
        state = np.stack((board == 1, board == -1, board == 0)).astype(np.float32)
        # state = np.expand_dims(state, axis=0)
        return state

    def get_decoded_policy(self, policy, board):
        """
        Decode the policy from the neural network to a format that can be used by the game.
        :return: The decoded policy.
        """
        mask = np.zeros([self.board_size] * 2)
        for col, row in board.legal_moves():
            mask[row][col] = 1
        mask = mask.flatten()
        # softmax:
        policy = np.exp(policy) / np.sum(np.exp(policy))
        policy = policy * mask
        policy = policy / np.sum(policy)
        return policy.reshape(self.board_size, self.board_size)

    def create_training_data(self):
        """
        Create training data for the neural network.
        :return: training data.
        """
        hex_game = Hex.HexBoard(self.board_size)
        memory_boards = []
        memory_pi = []
        while hex_game.check_outcome() == hex_game.ONGOING:
            board_state = np.copy(hex_game.board)
            memory_boards.append(board_state)
            root = Mcts.MCTS(Hex.HexBoard(self.board_size), iterations=10000).build_tree()
            pi = np.array(self.board_size ** 2 * [0])
            for child in root.children:
                pi[child.move[0] * self.board_size + child.move[1]] = child.visits
            memory_pi.append(np.exp(pi) / np.sum(np.exp(pi)))
            random_move = hex_game.legal_moves()[np.random.randint(0, len(hex_game.legal_moves()))]
            hex_game.make_move(random_move)
            hex_game = hex_game
            print(hex_game)
        z = hex_game.check_outcome()
        if z == hex_game.WHITE:
            z = 0
        elif z == hex_game.EMPTY:
            z = 0.5
        training_data = []
        memory_boards = memory_boards[::-1]
        memory_pi = memory_pi[::-1]
        for pi, board in zip(memory_pi, memory_boards):
            state = self.get_encoded_board(board)
            training_data.append([state, z, pi])
            z = 1 - z
        return training_data

    def train(self, num_of_epochs):
        """
        Train the neural network.
        :param num_of_epochs: number of random games to play.
        :param board_size: the size of the board.
        """
        training_data = []
        for _ in range(num_of_epochs):
            print("iteration: ", _ + 1, " out of ", num_of_epochs)
            training_data += self.create_training_data()
        print (training_data)
        print(training_data.__len__())
        states = np.array([x[0] for x in training_data])
        z = np.array([x[1] for x in training_data])
        PI = np.array([x[2] for x in training_data])
        print(states.shape)
        self.model.fit(states, [z, PI], epochs=20, batch_size=32)
        self.model.save_weights("model.weights.h5")

    def fetch_weights(self):
        """
        Fetch the weights of the model.
        :return: The weights of the model.
        """
        self.model.load_weights("model.weights.h5")

    def get_model(self):
        """
        Get the model.
        :return: The model.
        """
        return self.model


def main():
    """
    example of using NeuralNetwork class.
    :return:
    """
    board_size = 9
    # hex_game = Hex.HexBoard(board_size)
    # hex_game.make_move((4, 4))
    # hex_game.make_move((4, 5))
    # hex_game.make_move((5, 4))
    # print(hex_game)
    neural_network = NeuralNetwork(board_size)
    # state = neural_network.get_encoded_board(hex_game.board)
    # value, policy = neural_network.model.predict(state)
    # print("=====================================value=====================================")
    # print(value)
    # print("=====================================policy=====================================")
    # print(policy)
    # policy = neural_network.get_decoded_policy(policy, hex_game)
    # print("=====================================policy=====================================")
    # print(policy.reshape(9, 9))
    # neural_network.train(1000)
    neural_network.train(10)


# def train(self, num_of_epochs, board_size, batch_size, data, learning_rate=0.01):
#     for _ in range(num_of_epochs):
#         hex_game = Hex.HexBoard(board_size)
#         while hex_game.check_outcome() == hex_game.ONGOING:
#             mcts = Mcts.MCTS(hex_game, iterations=10000)
#             move = mcts.choose_move()
#             hex_game.make_move(move)
#         z = hex_game.check_outcome()
#         # train the neural network
#         # update the weights
#         # backpropagation
#
#
# def main():
#     """
#     simple example of using MCTS for Hex.
#     """
#     q = Queue()
#     mcts = Mcts.MCTS(Hex.HexBoard(9), iterations=5000)
#     hex_game = Hex.HexBoard(9)
#     game_screen = GameScreen.Screen(hex_game.board, 20, "red", "blue", q)
#     t1 = Thread(target=game_screen.run)
#     t1.start()
#
#     while hex_game.check_outcome() == hex_game.ONGOING:
#         if hex_game.current_player == Hex.HexBoard.BLACK:
#             mcts.board = hex_game
#             move = mcts.choose_move()
#         else:
#             move = q.get(True, None)
#         hex_game.make_move(move)
#         time.sleep(0.2)  # sleep for visual effect
#     time.sleep(1)
#     if hex_game.check_outcome() == hex_game.BLACK:
#         game_screen.winner = "RED"
#         print("Red wins")
#     else:
#         game_screen.winner = "BLUE"
#         print("Blue wins")
#     game_screen.game_ends = True
#
#     t1.join()


if __name__ == "__main__":
    print("Running example of MCTS for Hex.")
    main()
