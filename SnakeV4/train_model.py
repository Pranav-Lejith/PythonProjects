import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Game constants
WIDTH = 400
HEIGHT = 400
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE
SPEED = 100  # Increased speed for faster training

# Colors (not used in headless mode, but kept for consistency)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

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

# Initialize the model, optimizer, and loss function
model = SnakeNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Snake class
class Snake:
    def __init__(self):
        self.length = 1
        self.positions = [((GRID_WIDTH // 2), (GRID_HEIGHT // 2))]
        self.direction = random.choice([0, 1, 2, 3])
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
        self.randomize_position()

    def randomize_position(self):
        self.position = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))

# Game state
snake = Snake()
food = Food()

# Training loop
num_games = 1000
max_steps = 1000
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01

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

    # Check if there's a body part at each point
    body_l = point_l in snake.positions[1:]
    body_r = point_r in snake.positions[1:]
    body_u = point_u in snake.positions[1:]
    body_d = point_d in snake.positions[1:]

    state = [
        int(dir_r and body_r) or int(dir_l and body_l) or int(dir_u and body_u) or int(dir_d and body_d),
        int(dir_u and body_r) or int(dir_d and body_l) or int(dir_l and body_u) or int(dir_r and body_d),
        int(dir_d and body_r) or int(dir_u and body_l) or int(dir_r and body_u) or int(dir_l and body_d),
        int(dir_l),
        int(dir_r),
        int(dir_u),
        int(dir_d),
        int(food.position[0] < head[0]),
        int(food.position[0] > head[0]),
        int(food.position[1] < head[1]),
        int(food.position[1] > head[1])
    ]

    return torch.FloatTensor(state)

for game in range(num_games):
    snake.reset()
    food.randomize_position()
    game_over = False
    state = get_state(snake, food)

    for step in range(max_steps):
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, 2)
        else:
            with torch.no_grad():
                q_values = model(state)
                action = torch.argmax(q_values).item()

        # Move the snake
        snake.move(action)

        # Check for collision
        if snake.get_head_position() == food.position:
            snake.length += 1
            snake.score += 1
            food.randomize_position()
        elif snake.get_head_position() in snake.positions[1:]:
            game_over = True

        # Get new state
        next_state = get_state(snake, food)

        # Calculate reward
        reward = 1 if snake.get_head_position() == food.position else -1 if game_over else 0

        # Train the model
        with torch.no_grad():
            target = reward + (0 if game_over else 0.99 * torch.max(model(next_state)))
        current = model(state)[action]
        loss = criterion(current.unsqueeze(0), target.unsqueeze(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

        if game_over:
            break

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    print(f"Game {game + 1}/{num_games}, Score: {snake.score}, Epsilon: {epsilon:.2f}")

# Save the trained model
torch.save(model.state_dict(), "snake_ai_model.pth")

print("Training complete. Model saved as 'snake_ai_model.pth'")