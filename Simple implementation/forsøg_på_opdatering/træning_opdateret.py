import pygame
import torch
import numpy as np
from collections import deque
import random
import os
import matplotlib.pyplot as plt
import time

def train_flappy(env_class, agent_class):
    # Core training settings
    EPISODES = 40000
    BATCH_SIZE = 128
    BUFFER_CAPACITY = 10000
    WINDOW_SIZE = 100
    TARGET_UPDATE_FREQ = 50
    
    # Create save directory
    os.makedirs("models", exist_ok=True)
    model_path = "models/best_model.pth"
    
    # Initialize environment and agent
    env = env_class()
    agent = agent_class(state_size=5, action_size=2, hidden_size=256)
    replay_buffer = deque(maxlen=BUFFER_CAPACITY)
    
    # Training metrics
    window_scores = deque(maxlen=WINDOW_SIZE)
    all_scores = []
    moving_averages = []
    losses = []
    best_avg_score = float('-inf')
    start_time = time.time()
    
    # Set up the plots
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    plt.subplots_adjust(hspace=0.3)
    
    try:
        for episode in range(EPISODES):
            state = env.reset()
            done = False
            episode_losses = []
            
            while not done:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                
                replay_buffer.append((state, action, reward, next_state, done))
                
                if len(replay_buffer) >= BATCH_SIZE:
                    batch = random.sample(replay_buffer, BATCH_SIZE)
                    states, actions, rewards, next_states, dones = zip(*batch)
                    loss = agent.train_on_batch(
                        np.array(states),
                        np.array(actions),
                        np.array(rewards),
                        np.array(next_states),
                        np.array(dones)
                    )
                    episode_losses.append(loss)
                
                state = next_state
            
            # Post-episode processing
            window_scores.append(env.score)
            all_scores.append(env.score)
            if episode_losses:
                losses.append(np.mean(episode_losses))
            
            # Update target network
            if episode % TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()
            
            # Performance tracking and model saving
            if len(window_scores) == WINDOW_SIZE:
                current_avg = sum(window_scores) / WINDOW_SIZE
                moving_averages.append(current_avg)
                
                if current_avg > best_avg_score:
                    best_avg_score = current_avg
                    agent.save_model(model_path)
                    
                    # Calculate training metrics
                    elapsed_time = time.time() - start_time
                    episodes_per_minute = (episode + 1) / (elapsed_time / 60)
                    current_loss = np.mean(losses[-WINDOW_SIZE:]) if losses else 0
                    
                    print("\n" + "="*50)
                    print(f"Episode {episode + 1}")
                    print(f"New best {WINDOW_SIZE}-episode average: {current_avg:.2f}")
                    print(f"Training Speed: {episodes_per_minute:.1f} episodes/minute")
                    print(f"Time Elapsed: {elapsed_time/60:.1f} minutes")
                    print(f"Epsilon: {agent.epsilon:.3f}")
                    print(f"Average Loss: {current_loss:.6f}")
                    print("="*50)
            
            # Update visualization every 100 episodes
            if episode % 100 == 0 and episode > 0:
                # Update score plot
                ax1.clear()
                ax1.plot(all_scores[-1000:], 'b.', alpha=0.2, label='Scores')
                if moving_averages:
                    ax1.plot(range(len(moving_averages)), moving_averages, 'r-', label='Moving Average')
                ax1.grid(True, alpha=0.3)
                ax1.set_title(f'Training Progress (Episode {episode+1})')
                ax1.set_ylabel('Score')
                ax1.legend()
                
                # Update loss plot
                ax2.clear()
                if losses:
                    ax2.plot(losses[-1000:], 'g-', alpha=0.8, label='Loss')
                ax2.grid(True, alpha=0.3)
                ax2.set_title('Training Loss')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Loss')
                ax2.legend()
                
                plt.pause(0.01)
            
            # Render occasionally
            if episode % 1000 == 0:
                env.render()
                pygame.event.pump()
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        env.close()
        pygame.quit()
        plt.ioff()
        plt.close()

if __name__ == "__main__":
    from spil_opdatering_v2 import FlappyBird
    from agent_opdateret_v2 import DQNAgent
    
    train_flappy(env_class=FlappyBird, agent_class=DQNAgent)