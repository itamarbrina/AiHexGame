# Example file showing a circle moving on screen
from math import sqrt
import threading
import pygame as pg
import time
import UI.Hexagon as Hexagon


class Screen:
    """
    Class that represents the screen of the game.
    """
    def __init__(self, board, hex_size=20, first_player_color="red", second_player_color="blue"):
        """
        Initialize the Screen class.
        :param board: The board of the game.
        :param hex_size: The size of the hexagons.
        :param first_player_color: The color of the first player.
        :param second_player_color: The color of the second player.
        """
        self.update_board_flag = True
        self.hex_size = hex_size
        self.first_player_color = first_player_color
        self.second_player_color = second_player_color
        self.board = board
        self.surface = None
        self.matrix_of_hexagons = None

    def create_hexagons_grid(self):
        """
        Create a matrix of hexagons.
        :return: A matrix of hexagons.
        """
        matrix_of_hexagons = [[None for _ in range(len(self.board))] for _ in range(len(self.board))]
        starting_pos = pg.Vector2(self.surface.get_width() / 2,
                                  self.surface.get_height() / 2 - len(self.board) * self.hex_size * sqrt(3) / 2)
        point_in_matrix = [0, 0]
        counter_of_elements = 0
        reach_diagonal = False
        num_of_elements_in_row = 1
        for row in range(len(self.board) * 2 - 1):
            for col in range(num_of_elements_in_row):
                counter_of_elements += 1
                # Calculate the center position of each hexagon
                center_x = starting_pos.x + (col - num_of_elements_in_row / 2) * self.hex_size * 3
                center_y = starting_pos.y + row * self.hex_size * sqrt(3) / 2

                hexagon = Hexagon.Hexagon(self.surface, pg.Color("white"), pg.Color("black"), (center_x, center_y),
                                          self.hex_size, (point_in_matrix[0], point_in_matrix[1]))
                matrix_of_hexagons[point_in_matrix[0]][point_in_matrix[1]] = hexagon

                if point_in_matrix[0] == 0 and not reach_diagonal:
                    point_in_matrix[0] = point_in_matrix[1] + 1
                    point_in_matrix[1] = 0
                elif point_in_matrix[0] == len(self.board) - num_of_elements_in_row and reach_diagonal:
                    point_in_matrix[0] = len(self.board) - 1
                    point_in_matrix[1] = len(self.board) - num_of_elements_in_row + 1
                else:
                    point_in_matrix[0] -= 1
                    point_in_matrix[1] += 1
                if point_in_matrix[1] == len(self.board) - 1:
                    reach_diagonal = True

            if row >= len(self.board) - 1:
                num_of_elements_in_row -= 1
            else:
                num_of_elements_in_row += 1
        return matrix_of_hexagons

    def render_board(self):
        """
        Render the board on the screen.
        """
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                if self.board[i][j] == 1:
                    self.matrix_of_hexagons[i][j].col_in = pg.Color(self.first_player_color)
                elif self.board[i][j] == -1:
                    self.matrix_of_hexagons[i][j].col_in = pg.Color(self.second_player_color)
                self.matrix_of_hexagons[i][j].draw()

    def run(self):
        """
        Run the game.
        """
        pg.init()
        screen = pg.display.set_mode((1280, 720))
        self.surface = screen
        pg.display.set_caption("Hexagon Game")
        self.matrix_of_hexagons = self.create_hexagons_grid()
        self.surface.fill(pg.Color("white"))
        running = True
        while running:
            self.surface.fill(pg.Color("white"))
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
            self.render_board()
            pg.display.flip()  # Update the display
            pg.time.Clock().tick(60)  # Cap the frame rate
        pg.quit()


# Example of how to use the Screen class
def main():
    """
    Example of how to use the Screen class.
    """
    board_example = [[0 for _ in range(10)] for _ in range(10)]
    screen = Screen(board_example)
    screen_thread = threading.Thread(target=screen.run)
    screen_thread.start()
    for i in range(10):
        for j in range(10):
            if (i * 10 + j) % 2 == 0:
                board_example[i][j] = -1
            else:
                board_example[i][j] = 1
            screen.board = board_example
            time.sleep(0.3)
    screen_thread.join()


if __name__ == "__main__":
    main()
