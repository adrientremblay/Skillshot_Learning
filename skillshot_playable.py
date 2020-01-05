import pygame

from InputHandler import InputHandler
from SkillshotGame import SkillshotGame

# Colours
colours = [(0, 0, 0),
           (200, 100, 100),
           (100, 200, 100),
           (255, 0, 0),
           (0, 255, 0)]

pygame.init()

size = (520, 520)
screen = pygame.display.set_mode(size)
screen.fill((0, 100, 100))
pygame.display.set_caption("Skillshot Playable")

clock = pygame.time.Clock()

# Create surface
pixel_size = 2
game_board_size = [i * pixel_size for i in (250, 250)]
surface_board = pygame.Surface(game_board_size)

# Create game object
skillshotGame = SkillshotGame()

# Create inputHandler object
inputHandler = InputHandler()

run = True
while run:
    # Get and process events, including keypress
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        elif event.type == pygame.KEYDOWN:
            inputHandler.input_start(event.key)
        elif event.type == pygame.KEYUP:
            inputHandler.input_stop(event.key)
        if event.type == pygame.KEYDOWN:                # DEBUG
            if event.key == pygame.K_0:
                skillshotGame.game_reset()
                print("game reset")

    # do the actions in the inputHandler
    for player_inputs, player in zip(inputHandler.get_inputs(), (skillshotGame.player1, skillshotGame.player2)):
        if player_inputs.get("forwards"):
            player.move_forwards()
        if player_inputs.get("backwards"):
            player.move_backwards()
        if player_inputs.get("lookleft"):
            player.move_look_left()
        if player_inputs.get("lookright"):
            player.move_look_right()
        if player_inputs.get("shoot"):
            player.move_shoot_projectile()

    # Update game object status
    skillshotGame.game_tick()

    # Draw the combined board on the surface_board
    for index_y, row_x in enumerate(skillshotGame.get_board()):
        for index_x, item in enumerate(row_x):
            pixel = pygame.Rect((index_y * pixel_size, index_x * pixel_size), (pixel_size, pixel_size))
            pygame.draw.rect(surface_board, colours[item], pixel)

    # Update screen surface
    screen.blit(surface_board, (10, 10))
    pygame.display.flip()

    # Frame-rate limit
    clock.tick(60)
