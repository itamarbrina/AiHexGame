import keras
import numpy as np
import LOGIC.Puct as Puct
import LOGIC.Hex as Hex
import LOGIC.Mcts as Mcts
from tqdm import tqdm


def softmax(x, axis=-1):
    kw = dict(axis=axis, keepdims=True)
    xrel = x - x.max(**kw)
    exp_xrel = np.exp(xrel)
    return exp_xrel / exp_xrel.sum(**kw)


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
        self.value = keras.layers.Dense(1, activation='sigmoid')
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
        game_ended = False
        while not game_ended:
            if hex_game.check_outcome() != hex_game.ONGOING:
                game_ended = True
            board_state = np.copy(hex_game.board)
            memory_boards.append(board_state)
            root = Mcts.MCTS(hex_game, iterations=1500).build_tree()
            pi = np.array(self.board_size ** 2 * [0])
            for child in root.children:
                pi[child.move[0] * self.board_size + child.move[1]] = child.visits
            memory_pi.append(softmax(pi))
            if not game_ended:
                random_move = hex_game.legal_moves()[np.random.randint(0, len(hex_game.legal_moves()))]
                hex_game.make_move(random_move)
                # print(hex_game)
            hex_game = hex_game
        z = hex_game.check_outcome()
        if z == hex_game.WHITE:
            z = 0
        elif z == hex_game.DRAW:
            z = 0.5
        training_data = []
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
        for _ in tqdm(range(num_of_epochs), desc="Training", colour="green"):
            training_data += self.create_training_data()
        self.save_train_data(training_data)
        states = np.array([x[0] for x in training_data])
        z = np.array([x[1] for x in training_data])
        PI = np.array([x[2] for x in training_data])
        # print(states.shape)
        self.model.fit(states, [z, PI], epochs=1000, batch_size=32)
        accuracy = self.model.evaluate(states, [z, PI])
        print("accuracy: ", accuracy)
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.weights.h5")
        print("Saved model to disk")

    def fetch_model(self):
        """
        Fetch the weights of the model.
        :return: The weights of the model.
        """
        # self.model.load_weights("tmp.weights.h5")
        # self.model = keras.models.load_model("model.keras")
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("model.weights.h5")
        print("Loaded model from disk")

    def get_model(self):
        """
        Get the model.
        :return: The model.
        """
        return self.model

    def save_train_data(self, training_data):
        """
        Save the training data to a file.
        :param training_data: The training data.
        """
        with open("training_data.txt", "w") as file:
            for data in training_data:
                file.write(str(data) + "\n")
        # close the file
        file.close()


def main():
    """
    example of using NeuralNetwork class.
    :return:
    """
    board_size = 5
    neural_network = NeuralNetwork(board_size)
    neural_network.train(500)


if __name__ == "__main__":
    print("Running example of MCTS for Hex.")
    main()
