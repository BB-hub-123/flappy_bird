import pygame
import torch
import numpy as np
from collections import deque
import random
import os
import glob
import matplotlib.pyplot as plt
from datetime import datetime
import time

def train_flappy(env_class, agent_class):
    """
    Optimized training loop with improved file management and performance tracking
    """
    # Core training settings
    EPISODES = 40000
    BATCH_SIZE = 256  # Matched with agent's batch size
    BUFFER_CAPACITY = 5000
    WINDOW_SIZE = 100
    
    # Create save directory
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize environment and agent
    env = env_class()
    agent = agent_class(state_size=5, action_size=2, hidden_size=128)
    replay_buffer = deque(maxlen=BUFFER_CAPACITY)
    
    # Tracking variables
    window_scores = deque(maxlen=WINDOW_SIZE)
    all_scores = []
    moving_averages = []
    best_avg_score = float('-inf')
    start_time = time.time()
    
    # Set up the plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title('Flappy Bird DQN Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    
    try:
        for episode in range(EPISODES):
            episode_start = time.time()
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
            
            # Performance metrics
            elapsed_time = time.time() - start_time
            episodes_per_minute = (episode + 1) / (elapsed_time / 60)
            estimated_total_time = (EPISODES * elapsed_time) / (episode + 1)
            time_remaining = estimated_total_time - elapsed_time
            
            if len(window_scores) == WINDOW_SIZE:
                current_avg = sum(window_scores) / WINDOW_SIZE
                moving_averages.append(current_avg)
                
                # Update visualization every 50 episodes
                if episode % 500 == 0:
                    ax.clear()
                    ax.plot(all_scores[-1000:], 'b.', alpha=0.2, label='Scores')
                    ax.plot(range(len(moving_averages)), moving_averages, 'r-', label='Moving Average')
                    ax.grid(True, alpha=0.3)
                    ax.set_title(f'Training Progress (Episode {episode+1})')
                    ax.set_ylabel('Score')
                    ax.set_xlabel('Episode')
                    ax.legend()
                    plt.pause(0.01)
                
                # Save if we have a new best model
                if current_avg > best_avg_score:
                    best_avg_score = current_avg
                    
                    # Remove old model files
                    for old_model in glob.glob(os.path.join(save_dir, "*.pth")):
                        os.remove(old_model)
                    
                    # Save new best model
                    model_path = os.path.join(save_dir, f"best_model.pth")
                    agent.save_model(model_path)
                    
                    print("\n" + "="*50)
                    print(f"Episode {episode + 1}")
                    print(f"New best {WINDOW_SIZE}-episode average: {current_avg:.2f}")
                    print(f"Training Speed: {episodes_per_minute:.1f} episodes/minute")
                    print(f"Time Elapsed: {elapsed_time/60:.1f} minutes")
                    print(f"Estimated Time Remaining: {time_remaining/60:.1f} minutes")
                    print(f"Epsilon: {agent.epsilon:.3f}")
                    print("="*50)
            
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