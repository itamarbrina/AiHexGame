# Example file showing a circle moving on screen
from math import sqrt
import threading
import pygame as pg
import time


def draw_hex(surface, col_in, col_out, pos, a):
    x, y = pos
    points = [(x - a / 2, y - a * sqrt(3) / 2),
              (x + a / 2, y - a * sqrt(3) / 2),
              (x + a, y),
              (x + a / 2, y + a * sqrt(3) / 2),
              (x - a / 2, y + a * sqrt(3) / 2),
              (x - a, y)]
    pg.draw.polygon(surface, col_in, points)
    pg.draw.polygon(surface, col_out, points, 4)


class Screen:
    def __init__(self, board, hex_size=20, first_player_color="red", second_player_color="blue"):
        self.update_board_flag = True
        self.hex_size = hex_size
        self.first_player_color = first_player_color
        self.second_player_color = second_player_color
        self.board = board

    def render_board(self):

        screen = pg.display.set_mode((1280, 720))
        screen.fill(pg.Color("white"))

        starting_pos = pg.Vector2(screen.get_width() / 2,
                                  screen.get_height() / 2 - len(self.board) * self.hex_size * sqrt(3) / 2)
        point_in_matrix = [0, 0]
        counter_of_elements = 0
        # base = 0
        reach_diagonal = False
        num_of_elements_in_row = 1
        for row in range(len(self.board) * 2 - 1):
            for col in range(num_of_elements_in_row):
                counter_of_elements += 1
                # Calculate the center position of each hexagon
                center_x = starting_pos.x + (col - num_of_elements_in_row / 2) * self.hex_size * 3
                center_y = starting_pos.y + row * self.hex_size * sqrt(3) / 2

                if self.board[point_in_matrix[0]][point_in_matrix[1]] == 1:
                    draw_hex(screen, pg.Color(self.first_player_color), pg.Color("black"), (center_x, center_y),
                             self.hex_size)
                elif self.board[point_in_matrix[0]][point_in_matrix[1]] == -1:
                    draw_hex(screen, pg.Color(self.second_player_color), pg.Color("black"), (center_x, center_y),
                             self.hex_size)
                else:
                    draw_hex(screen, pg.Color("white"), pg.Color("black"), (center_x, center_y), self.hex_size)

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
        pg.display.flip()

    def update_board(self, board):
        self.board = board
        self.update_board_flag = True

    def run(self):
        pg.init()
        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
            if self.update_board_flag:
                self.render_board()
                self.update_board_flag = False
            pg.display.flip()  # Update the display
            pg.time.Clock().tick(60)  # Cap the frame rate
        pg.quit()


# Example of how to use the Screen class
def main():
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
            screen.update_board(board_example)
            time.sleep(0.3)
    screen_thread.join()


if __name__ == "__main__":
    main()
