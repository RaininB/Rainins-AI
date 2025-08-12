
import pygame
import random
import sys

pygame.init()

class Tetromino:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.shape = None

class I_Tetromino(Tetromino):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.shape = [[1, 1, 1, 1]]

class O_Tetromino(Tetromino):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.shape = [[1, 1], [1, 1]]

class J_Tetromino(Tetromino):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.shape = [[0, 0, 1],
                      [1, 1, 1],
                      [0, 0, 0]]

class L_Tetromino(Tetromino):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.shape = [[1, 1, 1],
                      [0, 0, 1],
                      [0, 0, 0]]

class S_Tetromino(Tetromino):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.shape = [[0, 1, 1],
                      [1, 1, 0],
                      [0, 0, 0]]

class Z_Tetromino(Tetromino):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.shape = [[1, 1, 0],
                      [0, 1, 1],
                      [0, 0, 0]]

class T_Tetromino(Tetromino):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.shape = [[0, 1, 0],
                      [1, 1, 1],
                      [0, 0, 0]]

class Tetris:
    def __init__(self):
        self.piece = None
        self.width = 400
        self.height = 500
        self.block_size = 30
        self.grid = [[0 for _ in range(self.width // self.block_size)] for _ in range(self.height // self.block_size)]
        self.score = 0

    def create_piece(self):
        piece_types = [I_Tetromino, O_Tetromino, J_Tetromino, L_Tetromino, S_Tetromino, Z_Tetromino, T_Tetromino]
        piece_type = random.choice(piece_types)
        self.piece = piece_type(4, 0)

    def draw_piece(self):
        for y, row in enumerate(self.piece.shape):
            for x, cell in enumerate(row):
                if cell == 1:
                    pygame.draw.rect(screen, (255, 255, 255), (self.piece.x * self.block_size + x * self.block_size,
                                                               self.piece.y * self.block_size + y * self.block_size,
                                                               self.block_size, self.block_size))

    def draw_grid(self):
        for y in range(len(self.grid)):
            for x in range(len(self.grid[y])):
                if self.grid[y][x] == 1:
                    pygame.draw.rect(screen, (255, 255, 255), ((x * self.block_size),
                                                               (y * self.block_size),
                                                               self.block_size, self.block_size))

    def move_piece_down(self):
        if self.piece.y + len(self.piece.shape) <= self.height // self.block_size:
            self.piece.y += 1
        else:
            for y, row in enumerate(self.piece.shape):
                for x, cell in enumerate(row):
                    if cell == 1:
                        self.grid[self.piece.y - y][self.piece.x + x] = 1

    def move_piece_left(self):
        if self.piece.x > 0 and all(cell == 1 for x, cell in enumerate(row) if x < len(row) // 2 for row in
                                   self.piece.shape):
            self.piece.x -= 1

    def move_piece_right(self):
        if self.piece.x + len(self.piece.shape[0]) <= self.width // self.block_size and all(cell == 1 for x, cell in
                                                                                             enumerate(row)
                                                                                             if x >= len(
                                                                                                 row) // 2 for row in
                                                                                             self.piece.shape):
            self.piece.x += 1

    def check_collision(self):
        for y, row in enumerate(self.piece.shape):
            for x, cell in enumerate(row):
                if cell == 1:
                    if (self.piece.y + y < len(self.grid) and self.piece.x + x < len(self.grid[0]) and
                            self.piece.y + y >= 0 and self.piece.x + x >= 0 and self.grid[self.piece.y + y][self.piece.x +
                                                                                                                x] == 1):
                        return True

    def main_loop(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.move_piece_left()
                    elif event.key == pygame.K_RIGHT:
                        self.move_piece_right()

            screen.fill((0, 0, 0))
            self.draw_grid()
            self.draw_piece()
            pygame.display.update()
            clock.tick(60)

            if self.check_collision():
                self.move_piece_down()
                for y, row in enumerate(self.piece.shape):
                    for x, cell in enumerate(row):
                        if cell == 1:
                            self.grid[self.piece.y - y][self.piece.x + x] = 1
                self.piece = None

        pygame.quit()

def main():
    global screen
    clock = pygame.time.Clock()
    tetris = Tetris()
    screen = pygame.display.set_mode((tetris.width, tetris.height))
    running = True
    while running:
        if not tetris.piece:
            tetris.create_piece()
        tetris.main_loop()

if __name__ == "__main__":
    main()
