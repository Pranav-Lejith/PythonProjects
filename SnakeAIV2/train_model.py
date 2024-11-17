import numpy as np
import pygame
import pickle
import random

# Initialize Pygame
pygame.init()

# Define constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
CELL_SIZE = 20
NUM_CELLS_X = SCREEN_WIDTH // CELL_SIZE
NUM_CELLS_Y = SCREEN_HEIGHT // CELL_SIZE
FPS = 35000

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Define actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# Number of state features
NUM_STATE_FEATURES = 8

# Initialize Q-table
with open('q_table.pkl', 'rb') as f:
    Q_TABLE = pickle.load(f)

# Define the game environment
class SnakeGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        self.snake = [(NUM_CELLS_X // 2, NUM_CELLS_Y // 2)]
        self.snake_dir = RIGHT
        self.food = self.place_food()
        self.done = False
    
    def place_food(self):
        return (random.randint(0, NUM_CELLS_X - 1), random.randint(0, NUM_CELLS_Y - 1))
    
    def get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # Create a state vector with relative positions and direction
        state = np.zeros(NUM_STATE_FEATURES)
        state[0] = (food_x - head_x) / NUM_CELLS_X
        state[1] = (food_y - head_y) / NUM_CELLS_Y
        state[2] = int(self.snake_dir == UP)
        state[3] = int(self.snake_dir == DOWN)
        state[4] = int(self.snake_dir == LEFT)
        state[5] = int(self.snake_dir == RIGHT)
        
        # Check for obstacles (wall or snake body)
        state[6] = int(head_x <= 0)
        state[7] = int(head_x >= NUM_CELLS_X - 1)
        
        return state
    
    def state_to_index(self, state):
        # Convert state to a single index
        state_idx = (
            int(state[0] * NUM_CELLS_X) +
            int(state[1] * NUM_CELLS_Y) * NUM_CELLS_X +
            int(state[2]) * NUM_CELLS_X * NUM_CELLS_Y +
            int(state[3]) * NUM_CELLS_X * NUM_CELLS_Y * 4 +
            int(state[4]) * NUM_CELLS_X * NUM_CELLS_Y * 4 * 4 +
            int(state[5]) * NUM_CELLS_X * NUM_CELLS_Y * 4 * 4 * 4 +
            int(state[6]) * NUM_CELLS_X * NUM_CELLS_Y * 4 * 4 * 4 * 2 +
            int(state[7]) * NUM_CELLS_X * NUM_CELLS_Y * 4 * 4 * 4 * 2 * 2
        )
        return state_idx
    
    def step(self, action):
        # Determine new direction based on action
        if action == UP:
            self.snake_dir = UP
        elif action == DOWN:
            self.snake_dir = DOWN
        elif action == LEFT:
            self.snake_dir = LEFT
        elif action == RIGHT:
            self.snake_dir = RIGHT
        
        head_x, head_y = self.snake[0]
        if self.snake_dir == UP:
            head_y -= 1
        elif self.snake_dir == DOWN:
            head_y += 1
        elif self.snake_dir == LEFT:
            head_x -= 1
        elif self.snake_dir == RIGHT:
            head_x += 1
        
        new_head = (head_x, head_y)
        if new_head in self.snake or head_x < 0 or head_y < 0 or head_x >= NUM_CELLS_X or head_y >= NUM_CELLS_Y:
            self.done = True
            reward = -1
        else:
            self.snake = [new_head] + self.snake[:-1]
            if new_head == self.food:
                self.snake.append(self.snake[-1])
                self.food = self.place_food()
                reward = 1
            else:
                reward = 0
        
        return reward, self.done

    def draw(self):
        self.screen.fill(BLACK)
        
        # Draw snake
        for segment in self.snake:
            pygame.draw.rect(self.screen, GREEN, pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        # Draw food
        pygame.draw.rect(self.screen, RED, pygame.Rect(self.food[0] * CELL_SIZE, self.food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        pygame.display.flip()

def run_snake():
    game = SnakeGame()
    while not game.done:
        state = game.get_state()
        state_idx = game.state_to_index(state)
        action = np.argmax(Q_TABLE[state_idx])
        
        reward, done = game.step(action)
        game.draw()
        game.clock.tick(FPS)
    
    print("Game Over")

if __name__ == "__main__":
    run_snake()
