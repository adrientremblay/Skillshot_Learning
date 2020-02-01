import pygame


class SkillshotGameDisplay(object):
    # Colours
    colours = [(0, 0, 0),
               (200, 100, 100),
               (100, 200, 100),
               (255, 0, 0),
               (0, 255, 0)]

    text_colours = [(0, 0, 0),
                    (100, 100, 100)]

    def __init__(self):
        pygame.init()

        self.size = (620, 520)
        self.screen = pygame.display.set_mode(self.size)
        self.screen.fill((0, 100, 100))
        pygame.display.set_caption("Skillshot Playable")

        self.clock = pygame.time.Clock()

        # Create surface
        self.pixel_size = 2
        self.game_board_size = [i * self.pixel_size for i in (250, 250)]
        self.surface_board = pygame.Surface(self.game_board_size)

        # Create font for text
        self.text_font = pygame.font.SysFont('Comic Sans MS', 12)

    def display_sequence(self, boards, epoch_number, frame=0):
        # takes list of boards for a single epoch and displays them
        # frame is the starting frame in the boards list
        # epoch_info is just to display the epoch info on the side

        boards_len = len(boards)

        run = True
        while run:
            # handle inputs
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

            # Draw the combined board on the surface_board
            for index_y, row_x in enumerate(boards[frame]):
                for index_x, item in enumerate(row_x):
                    pixel = pygame.Rect((index_y * self.pixel_size, index_x * self.pixel_size), (self.pixel_size, self.pixel_size))
                    pygame.draw.rect(self.surface_board, self.colours[item], pixel)

            # Prepare text info
            frame_info = self.text_font.render("Frame: {} / {}".format(frame, boards_len),
                                               True,
                                               self.text_colours[0],
                                               self.text_colours[1])
            epoch_info = self.text_font.render("Epoch: {}".format(epoch_number),
                                               True,
                                               self.text_colours[0],
                                               self.text_colours[1])

            # increment the frame
            frame += 1
            if frame >= boards_len:
                run = False

            # Update screen surface
            self.screen.blit(self.surface_board, (10, 10))
            self.screen.blit(frame_info, (250 * self.pixel_size + 20, 10))
            self.screen.blit(epoch_info, (250 * self.pixel_size + 20, 40))

            pygame.display.flip()

            # Frame-rate limit
            self.clock.tick(30)

    @staticmethod
    def close_window():
        # closes the pygame window
        pygame.display.quit()
        pygame.quit()