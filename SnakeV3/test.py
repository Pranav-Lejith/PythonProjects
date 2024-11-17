import pygame
import numpy as np
from snake_game import SnakeGame
from dqn_agent import DQNAgent

def test(model_path):
    pygame.init()
    env = SnakeGame()
    state_size = 11
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    agent.load(model_path)
    agent.epsilon = 0  # No exploration during testing
    
    screen = pygame.display.set_mode((env.width, env.height))
    clock = pygame.time.Clock()
    
    state = env.reset()
    done = False
    score = 0
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        state = next_state
        score += reward
        
        env.render(screen)
        pygame.display.flip()
        clock.tick(10)  # Adjust game speed here
    
    print(f"Game Over. Final Score: {score}")
    pygame.quit()

if __name__ == "__main__":
    test("snake_model_final.pth")