import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialize Pygame
pygame.init()

# Game constants
WIDTH = 400
HEIGHT = 400
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE
SPEED = 10  # Normal speed for testing

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# Initialize the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game AI - Test")

# Neural Network
class SnakeNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(11, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Load the trained model
model = SnakeNN()
model.load_state_dict(torch.load("snake_ai_model.pth"))
model.eval()

# Snake class
class Snake:
    def __init__(self):
        self.length = 1
        self.positions = [((GRID_WIDTH // 2), (GRID_HEIGHT // 2))]
        self.direction = random.choice([0, 1, 2, 3])
        self.color = GREEN
        self.score = 0

    def get_head_position(self):
        return self.positions[0]

    def move(self, action):
        cur = self.get_head_position()
        x, y = cur

        if action == 0:  # Straight
            pass
        elif action == 1:  # Right
            self.direction = (self.direction + 1) % 4
        elif action == 2:  # Left
            self.direction = (self.direction - 1) % 4

        if self.direction == 0:
            y -= 1
        elif self.direction == 1:
            x += 1
        elif self.direction == 2:
            y += 1
        elif self.direction == 3:
            x -= 1

        self.positions.insert(0, ((x % GRID_WIDTH), (y % GRID_HEIGHT)))
        if len(self.positions) > self.length:
            self.positions.pop()

    def reset(self):
        self.length = 1
        self.positions = [((GRID_WIDTH // 2), (GRID_HEIGHT // 2))]
        self.direction = random.choice([0, 1, 2, 3])
        self.score = 0

# Food class
class Food:
    def __init__(self):
        self.position = (0, 0)
        self.color = RED
        self.randomize_position()

    def randomize_position(self):
        self.position = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))

# Game state
snake = Snake()
food = Food()

def get_state(snake, food):
    head = snake.get_head_position()
    point_l = ((head[0] - 1) % GRID_WIDTH, head[1])
    point_r = ((head[0] + 1) % GRID_WIDTH, head[1])
    point_u = (head[0], (head[1] - 1) % GRID_HEIGHT)
    point_d = (head[0], (head[1] + 1) % GRID_HEIGHT)

    dir_l = snake.direction == 3
    dir_r = snake.direction == 1
    dir_u = snake.direction == 0
    dir_d = snake.direction == 2

    state = [
        (dir_r and snake.positions[1] == point_r) or
        (dir_l and snake.positions[1] == point_l) or
        (dir_u and snake.positions[1] == point_u) or
        (dir_d and snake.positions[1] == point_d),

        (dir_u and snake.positions[1] == point_r) or
        (dir_d and snake.positions[1] == point_l) or
        (dir_l and snake.positions[1] == point_u) or
        (dir_r and snake.positions[1] == point_d),

        (dir_d and snake.positions[1] == point_r) or
        (dir_u and snake.positions[1] == point_l) or
        (dir_r and snake.positions[1] == point_u) or
        (dir_l and snake.positions[1] == point_d),

        dir_l,
        dir_r,
        dir_u,
        dir_d,

        food.position[0] < head[0],
        food.position[0] > head[0],
        food.position[1] < head[1],
        food.position[1] > head[1]
    ]

    return list(map(int, state))

# Game loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get the current state
    state = get_state(snake, food)

    # Get the action from the trained model
    with torch.no_grad():
        q_values = model(torch.FloatTensor(state))
        action = torch.argmax(q_values).item()

    # Move the snake
    snake.move(action)

    # Check for collision
    if snake.get_head_position() == food.position:
        snake.length += 1
        snake.score += 1
        food.randomize_position()
    elif snake.get_head_position() in snake.positions[1:]:
        snake.reset()
        food.randomize_position()

    # Update the display
    screen.fill(BLACK)
    for pos in snake.positions:
        pygame.draw.rect(screen, GREEN, (pos[0] * GRID_SIZE, pos[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
    pygame.draw.rect(screen, RED, (food.position[0] * GRID_SIZE, food.position[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
    
    # Display the score
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {snake.score}", True, WHITE)
    screen.blit(score_text, (10, 10))
    
    pygame.display.flip()

    # Control the game speed
    clock.tick(SPEED)

pygame.quit()