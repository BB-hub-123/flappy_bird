import pygame
import torch
import numpy as np
from collections import deque
import random
import os
import glob
import matplotlib.pyplot as plt
from datetime import datetime

def train_flappy(env_class, agent_class):
    """
    Optimized training loop with improved file management and performance
    """
    # Core training settings
    EPISODES = 40000
    BATCH_SIZE = 64
    BUFFER_CAPACITY = 10000  # Smaller buffer for better memory usage
    WINDOW_SIZE = 100
    
    # Create save directory
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize environment and agent
    env = env_class()
    agent = agent_class(state_size=5, action_size=2, hidden_size=64)
    replay_buffer = deque(maxlen=BUFFER_CAPACITY)
    
    # Tracking variables
    window_scores = deque(maxlen=WINDOW_SIZE)
    all_scores = []  # Keep track of all scores for visualization
    moving_averages = []  # Keep track of moving averages
    best_avg_score = float('-inf')
    
    # Set up the plot
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title('Flappy Bird DQN Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    
    try:
        for episode in range(EPISODES):
            state = env.reset()
            episode_reward = 0
            done = False
            
            # Episode loop
            while not done:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                episode_reward += reward
                
                replay_buffer.append((state, action, reward, next_state, done))
                
                if len(replay_buffer) >= BATCH_SIZE:
                    # Sample and train less frequently
                    if len(replay_buffer) % 4 == 0:
                        batch = random.sample(replay_buffer, BATCH_SIZE)
                        states, actions, rewards, next_states, dones = zip(*batch)
                        agent.train_on_batch(np.array(states), 
                                           np.array(actions), 
                                           np.array(rewards), 
                                           np.array(next_states), 
                                           np.array(dones))
                
                state = next_state
            
            # Post-episode processing
            window_scores.append(env.score)
            all_scores.append(env.score)
            
            # Only update when we have enough episodes for meaningful average
            if len(window_scores) == WINDOW_SIZE:
                current_avg = sum(window_scores) / WINDOW_SIZE
                moving_averages.append(current_avg)
                
                # Update visualization every 50 episodes to maintain performance
                if episode % 50 == 0:
                    ax.clear()
                    ax.plot(all_scores[-1000:], 'b.', alpha=0.2, label='Scores')
                    ax.plot(range(len(moving_averages)), moving_averages, 'r-', label='Moving Average')
                    ax.grid(True, alpha=0.3)
                    ax.set_title(f'Training Progress (Episode {episode+1})')
                    ax.set_ylabel('Score')
                    ax.set_xlabel('Episode')
                    ax.legend()
                    plt.pause(0.01)  # Short pause to update display
                current_avg = sum(window_scores) / WINDOW_SIZE
                
                # Save only if we have a new best model
                if current_avg > best_avg_score:
                    best_avg_score = current_avg
                    
                    # Remove old model files
                    for old_model in glob.glob(os.path.join(save_dir, "*.pth")):
                        os.remove(old_model)
                    
                    # Save new best model
                    model_path = os.path.join(save_dir, f"best_model.pth")
                    agent.save_model(model_path)
                    
                    print(f"\nEpisode {episode + 1}")
                    print(f"New best {WINDOW_SIZE}-episode average: {current_avg:.2f}")
                    print(f"Epsilon: {agent.epsilon:.3f}")
            
            # Update target network periodically
            if episode % 100 == 0:
                agent.update_target_network()
            
            # Minimal rendering
            if episode % 1000 == 0:
                env.render()
                pygame.event.pump()
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        env.close()
        pygame.quit()

if __name__ == "__main__":
    from spil_opdatering import FlappyBird
    from agent_opdateret_v2 import DQNAgent
    
    train_flappy(env_class=FlappyBird, agent_class=DQNAgent)