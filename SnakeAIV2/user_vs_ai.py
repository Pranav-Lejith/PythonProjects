import pygame
import torch
from train_model import SnakeGameAI, DQN

def user_vs_ai():
    game = SnakeGameAI()
    model = DQN()
    model.load_state_dict(torch.load('snake_model.pth'))  # Load the trained model

    clock = pygame.time.Clock()
    game_over = False
    user_action = (0, -1)  # Initial direction for the user
    state = game.reset()

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    user_action = (-1, 0)
                elif event.key == pygame.K_RIGHT:
                    user_action = (1, 0)
                elif event.key == pygame.K_UP:
                    user_action = (0, -1)
                elif event.key == pygame.K_DOWN:
                    user_action = (0, 1)

        q_values = model(torch.FloatTensor(state))
        ai_action = q_values.argmax().item()

        # Perform AI and User moves
        _, _, game_over = game.step(user_action)
        _, _, game_over = game.step(ai_action)

        # Render game
        game.render()
        clock.tick(10)

if __name__ == "__main__":
    user_vs_ai()
