import pygame as pg
from math import sqrt


class Hexagon:
    """
    Class that represents a hexagon.
    """

    def __init__(self, surface, col_in, col_out, pos, radius, position_in_matrix=(0, 0)):
        """
        Initialize the Hexagon class.
        :param surface: The surface (screen) to draw the hexagon on.
        :param col_in: The color of the hexagon.
        :param col_out: The color of the border of the hexagon.
        :param pos: The position of the center of the hexagon.
        :param radius: The radius (size) of the hexagon.
        :param position_in_matrix: The position of the hexagon in the matrix.
        """
        self.surface = surface
        self.col_in = col_in
        self.col_out = col_out
        self.pos = pos
        self.radius = radius
        self.position_in_matrix = position_in_matrix
        self.clicked = False

    def draw(self):
        """
        Draw the hexagon on the screen.
        :return:  True if the mouse is clicked the hexagon.
        """
        action = False

        x, y = self.pos
        points = [(x - self.radius / 2, y - self.radius * sqrt(3) / 2),
                  (x + self.radius / 2, y - self.radius * sqrt(3) / 2),
                  (x + self.radius, y),
                  (x + self.radius / 2, y + self.radius * sqrt(3) / 2),
                  (x - self.radius / 2, y + self.radius * sqrt(3) / 2),
                  (x - self.radius, y)]
        pg.draw.polygon(self.surface, self.col_in, points)
        pg.draw.polygon(self.surface, self.col_out, points, 4)

        pos = pg.mouse.get_pos()
        if points[0][0] < pos[0] < points[1][0] and points[0][1] < pos[1] < points[3][1]:
            if pg.mouse.get_pressed()[0] and not self.clicked:
                self.clicked = True
                action = True
        if pg.mouse.get_pressed()[0] == 0:
            self.clicked = False
        return action
