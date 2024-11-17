import pygame
import random
import numpy as np

class SnakeGame:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = random.choice([(0, -1), (0, 1), (-1, 0), (1, 0)])
        self.food = self._place_food()
        self.score = 0
        self.game_over = False
        return self._get_state()

    def _place_food(self):
        while True:
            food = (random.randint(0, self.width // 20 - 1) * 20,
                    random.randint(0, self.height // 20 - 1) * 20)
            if food not in self.snake:
                return food

    def step(self, action):
        if self.game_over:
            return self._get_state(), 0, True

        # 0: straight, 1: right, 2: left
        if action == 1:
            self.direction = (self.direction[1], -self.direction[0])
        elif action == 2:
            self.direction = (-self.direction[1], self.direction[0])

        new_head = (self.snake[0][0] + self.direction[0] * 20,
                    self.snake[0][1] + self.direction[1] * 20)

        # Check if snake hits the wall or itself
        if (new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height or
            new_head in self.snake):
            self.game_over = True
            return self._get_state(), -10, True

        self.snake.insert(0, new_head)

        reward = 0
        if new_head == self.food:
            self.score += 1
            reward = 1
            self.food = self._place_food()
        else:
            self.snake.pop()

        return self._get_state(), reward, False

    def _get_state(self):
        head = self.snake[0]
        point_l = (head[0] - 20, head[1])
        point_r = (head[0] + 20, head[1])
        point_u = (head[0], head[1] - 20)
        point_d = (head[0], head[1] + 20)
        
        dir_l = self.direction == (-1, 0)
        dir_r = self.direction == (1, 0)
        dir_u = self.direction == (0, -1)
        dir_d = self.direction == (0, 1)

        state = [
            # Danger straight
            (dir_r and self._is_collision(point_r)) or 
            (dir_l and self._is_collision(point_l)) or 
            (dir_u and self._is_collision(point_u)) or 
            (dir_d and self._is_collision(point_d)),

            # Danger right
            (dir_u and self._is_collision(point_r)) or 
            (dir_d and self._is_collision(point_l)) or 
            (dir_l and self._is_collision(point_u)) or 
            (dir_r and self._is_collision(point_d)),

            # Danger left
            (dir_d and self._is_collision(point_r)) or 
            (dir_u and self._is_collision(point_l)) or 
            (dir_r and self._is_collision(point_u)) or 
            (dir_l and self._is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            self.food[0] < head[0],  # food left
            self.food[0] > head[0],  # food right
            self.food[1] < head[1],  # food up
            self.food[1] > head[1]   # food down
            ]

        return np.array(state, dtype=int)

    def _is_collision(self, pt):
        return (pt[0] < 0 or pt[0] >= self.width or
                pt[1] < 0 or pt[1] >= self.height or
                pt in self.snake)

    def render(self, surface):
        surface.fill((0, 0, 0))
        for segment in self.snake:
            pygame.draw.rect(surface, (0, 255, 0), (segment[0], segment[1], 20, 20))
        pygame.draw.rect(surface, (255, 0, 0), (self.food[0], self.food[1], 20, 20))