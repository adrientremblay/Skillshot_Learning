import pygame


class SkillshotGameDisplay(object):
    # Colours
    colours = [(0, 0, 0),
               (200, 100, 100),
               (100, 200, 100),
               (255, 0, 0),
               (0, 255, 0)]

    def __init__(self):
        pygame.init()

        self.size = (520, 520)
        self.screen = pygame.display.set_mode(self.size)
        self.screen.fill((0, 100, 100))
        pygame.display.set_caption("Skillshot Playable")

        self.clock = pygame.time.Clock()

        # Create surface
        self.pixel_size = 2
        self.game_board_size = [i * self.pixel_size for i in (250, 250)]
        self.surface_board = pygame.Surface(self.game_board_size)

    def display_sequence(self, boards):
        # takes list of boards for a single epoch and displays them
        # TODO change to use while loop and get new board every frame instead
        for board in boards:
            # Draw the combined board on the surface_board
            for index_y, row_x in enumerate(board):
                for index_x, item in enumerate(row_x):
                    pixel = pygame.Rect((index_y * self.pixel_size, index_x * self.pixel_size), (self.pixel_size, self.pixel_size))
                    pygame.draw.rect(self.surface_board, self.colours[item], pixel)

            # Update screen surface
            self.screen.blit(self.surface_board, (10, 10))
            pygame.display.flip()

            # Frame-rate limit
            self.clock.tick(30)
