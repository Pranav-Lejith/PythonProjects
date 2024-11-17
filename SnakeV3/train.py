import pygame
import numpy as np
from snake_game import SnakeGame
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt

def train():
    env = SnakeGame()
    state_size = 11
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    n_episodes = 1000
    
    scores = []
    
    for e in range(n_episodes):
        state = env.reset()
        score = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
        agent.update_target_model()
        scores.append(score)
        
        print(f"Episode: {e+1}/{n_episodes}, Score: {score}, Epsilon: {agent.epsilon:.2f}")
        
        if (e+1) % 100 == 0:
            agent.save(f"snake_model_episode_{e+1}.pth")
    
    agent.save("snake_model_final.pth")
    
    # Plot the scores
    plt.plot(scores)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig('training_progress.png')
    plt.show()

if __name__ == "__main__":
    train()