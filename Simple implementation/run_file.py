import pygame
import torch
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from simpel_spil import FlappyBird
from agent import DQNAgent

# Simple replay buffer 
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)

# Training settings
EPISODES = 10000
BATCH_SIZE = 62
BUFFER_CAPACITY = 10000
PRINT_INTERVAL = 50  # How often to print and plot

# Initialize everything
env = FlappyBird()
agent = DQNAgent(state_size=5, action_size=2, hidden_size=64)  # Fixed state_size to 5
replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

# Lists for tracking metrics
scores = []  # List to track scores
losses = []  # List to track losses
steps_per_episode = []  # List to track steps per episode
best_score = float('-inf')  # Keep track of the best score

# Training loop
try:
    for episode in range(EPISODES):
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        gradient_steps = 0
        done = False
        steps = 0
        
        while not done:
            # Get action from agent
            action = agent.act(state)
            
            # Take action in environment
            next_state, reward, done = env.step(action)
            
            # Store transition in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)
            
            # Train if enough samples
            if len(replay_buffer) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
                loss = agent.train_on_batch(states, actions, rewards, next_states, dones)
                episode_loss += loss
                gradient_steps += 1
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Render every nth episode
            if episode % 50 == 0:
                env.render()
                pygame.event.pump()
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
        
        # Update target network periodically
        if episode % agent.target_update == 0:
            agent.update_target_network()
        
        # Store episode metrics
        scores.append(episode_reward)
        steps_per_episode.append(steps)
        losses.append(episode_loss / (gradient_steps + 1) if gradient_steps > 0 else 0)
        
        # Save best model
        if episode_reward > best_score:
            best_score = episode_reward
            torch.save({
                'episode': episode,
                'model_state_dict': agent.policy_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'score': best_score,
                'epsilon': agent.epsilon
            }, 'flappy_best_model.pth')
        
        # Print and plot progress every PRINT_INTERVAL episodes
        if (episode + 1) % PRINT_INTERVAL == 0:
            # Calculate average score over the last PRINT_INTERVAL episodes
            avg_score = np.mean(scores[-PRINT_INTERVAL:]) if len(scores) >= PRINT_INTERVAL else np.mean(scores)
            avg_steps = np.mean(steps_per_episode[-PRINT_INTERVAL:]) if len(steps_per_episode) >= PRINT_INTERVAL else np.mean(steps_per_episode)
            print(f"Episode: {episode+1}, Avg Score: {avg_score:.2f}, Avg Steps: {avg_steps:.1f}, Epsilon: {agent.epsilon:.3f}")
            
            # Plot metrics
            plt.figure(1)
            plt.clf()  # Clear the previous plot
            plt.subplot(311)
            plt.plot(scores, '.')
            plt.title(f'Training Progress (ε={agent.epsilon:.3f})')
            plt.ylabel('Score')
            plt.grid(True)
            
            plt.subplot(312)
            plt.plot(steps_per_episode, '.')
            plt.ylabel('Steps')
            plt.grid(True)
            
            plt.subplot(313)
            plt.plot(losses, '.')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.grid(True)
            
            plt.pause(0.1)  # Pause to allow the plot to update

except KeyboardInterrupt:
    print("\nTraining interrupted by user")
finally:
    env.close()  # Close the environment and quit pygame
    pygame.quit()


