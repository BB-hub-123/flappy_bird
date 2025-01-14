import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import sys
import pygame
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output

from Rikke_agent import DQNAgent
from Rikke_test_env import FlappyBirdEnv
from replay_buffer import ReplayBuffer

def plot_training_results(episode_rewards, episode_lengths, window=100):
    """Plot the training progress."""
    if len(episode_rewards) < window or len(episode_lengths) < window:
        print("Not enough data to plot training results yet.")
        return

    plt.figure(figsize=(12, 4))
    
    plt.subplot(121)
    plt.title('Episode Rewards (Moving Average)')
    plt.plot(np.convolve(episode_rewards, np.ones(window)/window, mode='valid'))
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(122)
    plt.title('Episode Lengths (Moving Average)')
    plt.plot(np.convolve(episode_lengths, np.ones(window)/window, mode='valid'))
    plt.xlabel('Episode')
    plt.ylabel('Length')
    
    plt.tight_layout()
    plt.show()

def train_dqn(buffer_capacity=100000, save_frequency=1000):
    # Initialize environment and agent
    env = FlappyBirdEnv(decision_frequency=4, speed_multiplier=2)
    state_size = 5  # Based on your environment's state space
    action_size = 2  # Flap or don't flap
    agent = DQNAgent(state_size=state_size, 
                    action_size=action_size,
                    hidden_size=64)
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    
    # Training parameters
    num_episodes = 1000
    max_steps = 1000
    
    # Tracking metrics
    episode_rewards = []
    episode_lengths = []
    best_reward = float('-inf')
    
    try:
        # Training loop
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(max_steps):
                # Select action
                action = agent.act(state)
                
                # Take action in environment
                next_state, reward, done, info = env.step(action)
                
                # Store experience in replay buffer
                replay_buffer.push(state, action, reward, next_state, done)
                
                # Train the network
                if len(replay_buffer) >= agent.batch_size:
                    states, actions, rewards, next_states, dones = replay_buffer.sample(agent.batch_size)
                    agent.train_on_batch(states, actions, rewards, next_states, dones)
                
                # Update metrics
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            # Track progress
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save({
                    'episode': episode,
                    'model_state_dict': agent.policy_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'reward': best_reward,
                }, 'best_flappy_model.pth')
            
            # Save replay buffer periodically
            if len(replay_buffer) > 0 and episode > 0 and episode % save_frequency == 0:
                replay_buffer.save(episode)
            
            # Print progress
            if (episode + 1) % 100 == 0:
                if len(episode_rewards) > 0:
                    avg_reward = np.mean(episode_rewards[-100:])
                else:
                    avg_reward = 0

                if len(episode_lengths) > 0:
                    avg_length = np.mean(episode_lengths[-100:])
                else:
                    avg_length = 0

                print(f"Episode: {episode + 1}/{num_episodes}")
                print(f"Average Reward (last 100): {avg_reward:.2f}")
                print(f"Average Length (last 100): {avg_length:.2f}")
                print(f"Epsilon: {agent.epsilon:.3f}")
                print(f"Replay Buffer Size: {len(replay_buffer)}")
                print("----------------------------------------")
                
                # Plot progress
                plot_training_results(episode_rewards, episode_lengths)
        
        print("Training complete.")
        return agent, episode_rewards, episode_lengths, replay_buffer

    except Exception as e:
        print(f"Error during training: {e}")
        raise e
    finally:
        env.close()

def test_agent(agent, num_episodes=5):
    """Test the trained agent."""
    env = FlappyBirdEnv(decision_frequency=4, speed_multiplier=1)  # Normal speed for testing
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Handle pygame events to keep the window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
        
        print(f"Test Episode {episode + 1} Score: {total_reward}")

if __name__ == "__main__":
    try:
        # Set random seeds for reproducibility
        random.seed(42) 
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Train the agent
        trained_agent, rewards_history, length_history, replay_buffer = train_dqn()
        
        # Plot final training results
        plot_training_results(rewards_history, length_history)
        
        # Test the trained agent
        test_agent(trained_agent)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
    finally:
        pygame.quit()
        sys.exit()
