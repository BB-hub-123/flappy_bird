import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import sys
import pygame
from collections import deque
import matplotlib.pyplot as plt

from brors_agent_v2 import DQNAgent  # Updated import
from env_med_render_knap import FlappyBirdEnv

# Simple replay buffer implementation to ensure compatibility
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)

def plot_training_metrics(episode_rewards, losses, epsilons, window=10):
    """Plot training metrics including rewards, loss, and epsilon."""
    try:
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Moving average of rewards
        plt.subplot(3, 1, 1)
        episodes = np.arange(len(episode_rewards))
        if len(episode_rewards) >= window:
            reward_ma = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            ma_episodes = episodes[window-1:]
            plt.plot(ma_episodes, reward_ma)
        else:
            plt.plot(episodes, episode_rewards)
        plt.title('Average Reward (Moving Average)')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        # Plot 2: Loss over time
        plt.subplot(3, 1, 2)
        if losses:
            plt.plot(np.arange(len(losses)), losses)
        plt.title('Training Loss')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Plot 3: Epsilon over time
        plt.subplot(3, 1, 3)
        plt.plot(np.arange(len(epsilons)), epsilons)
        plt.title('Epsilon Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'training_metrics_{len(episode_rewards)}.png')
        plt.close()
        
    except Exception as e:
        print(f"Error in plotting: {e}")

def train_dqn(buffer_capacity=10000):
    print("Starting training...")
    
    try:
        # Initialize environment and agent
        env = FlappyBirdEnv(decision_frequency=4, speed_multiplier=4, render=False)
        state_size = 7
        action_size = 2
        agent = DQNAgent(state_size=state_size, 
                        action_size=action_size,
                        hidden_size=64)
        
        print(f"Environment and agent initialized. Using device: {agent.device}")
        
        # Initialize replay buffer
        replay_buffer = ReplayBuffer(buffer_capacity)
        print("Replay buffer initialized")
        
        # Training parameters
        num_episodes = 2000
        max_steps = 500
        print_frequency = 100  # Print every episode
        plot_frequency = 1000  # Plot every 10 episodes
        
        # Tracking metrics
        episode_rewards = []
        losses = []
        epsilons = []
        best_reward = float('-inf')
        
        print("Starting training loop...")
        # Training loop
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_losses = []
            
            for step in range(max_steps):
                # Select and perform action
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                replay_buffer.push(state, action, reward, next_state, done)
                
                # Train if enough samples
                if len(replay_buffer) >= agent.batch_size:
                    states, actions, rewards, next_states, dones = replay_buffer.sample(agent.batch_size)
                    loss = agent.train_on_batch(states, actions, rewards, next_states, dones)
                    episode_losses.append(loss)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Update target network
            if episode % agent.target_update == 0:
                agent.update_target_network()
            
            # Track metrics
            episode_rewards.append(episode_reward)
            if episode_losses:
                losses.append(np.mean(episode_losses))
            epsilons.append(agent.epsilon)
            
            # Print progress
            if episode % print_frequency == 0:
                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"Steps: {step + 1}")
                print(f"Reward: {episode_reward:.2f}")
                print(f"Epsilon: {agent.epsilon:.3f}")
                print(f"Buffer size: {len(replay_buffer)}")
                if episode_losses:
                    print(f"Average Loss: {np.mean(episode_losses):.4f}")
            
            # Plot progress
            if episode % plot_frequency == 0:
                plot_training_metrics(episode_rewards, losses, epsilons)
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save({
                    'episode': episode,
                    'model_state_dict': agent.policy_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'reward': best_reward,
                }, 'best_flappy_model.pth')
                print(f"\nNew best model saved with reward: {best_reward:.2f}")
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        env.close()
        
    return agent, episode_rewards, losses, epsilons, replay_buffer

if __name__ == "__main__":
    try:
        # Set random seeds
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Train the agent
        trained_agent, rewards_history, loss_history, epsilon_history, replay_buffer = train_dqn()
        
        # Save final plots
        plot_training_metrics(rewards_history, loss_history, epsilon_history)
        print("\nTraining completed successfully")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        pygame.quit()
        sys.exit()