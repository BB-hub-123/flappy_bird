import pygame
import torch
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        # Add with max priority for new experiences
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
        
    def sample(self, batch_size):
        # Prioritized sampling
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), p=probs)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5  # Small constant for stability
            
    def __len__(self):
        return len(self.buffer)

def train_flappy(env_class, agent_class):
    # Training settings
    EPISODES = 20000
    BATCH_SIZE = 64
    BUFFER_CAPACITY = 50000
    PRINT_INTERVAL = 25
    SAVE_INTERVAL = 100
    EVAL_INTERVAL = 500
    
    # Create training directory for saving models and logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"training_runs/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize environment, agent, and replay buffer
    env = env_class()
    agent = agent_class(state_size=5, action_size=2, hidden_size=128)
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)
    
    # Metrics tracking initialization
    scores = []
    losses = []
    avg_scores = []
    steps_per_episode = []
    best_avg_score = float('-inf')
    
    # Training history dictionary for comprehensive logging
    history = {
        'scores': [],
        'avg_scores': [],
        'losses': [],
        'steps': [],
        'epsilon': []
    }
    
    # Set up interactive plotting
    plt.ion()
    figure = plt.figure(figsize=(12, 8))
    
    # Main training loop
    try:
        for episode in range(EPISODES):
            state = env.reset()
            episode_reward = 0
            episode_loss = 0
            steps = 0
            
            # Episode loop
            while True:
                # Get action from agent
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                
                # Store experience in replay buffer
                replay_buffer.push(state, action, reward, next_state, done)
                
                # Training step if we have enough samples
                if len(replay_buffer) >= agent.batch_size:
                    states, actions, rewards, next_states, dones = replay_buffer.sample(agent.batch_size)
                    loss = agent.train_on_batch(states, actions, rewards, next_states, dones)
                    episode_loss += loss
                
                # Soft update target network
                if agent.training_steps % agent.target_update == 0:
                    agent.update_target_network()
                
                # Update state and metrics
                state = next_state
                episode_reward += reward
                steps += 1
                
                # Render every nth episode
                if episode % PRINT_INTERVAL == 0:
                    env.render()
                    pygame.event.pump()
                
                if done:
                    break
            
            # Post-episode updates
            scores.append(env.score)
            steps_per_episode.append(steps)
            losses.append(episode_loss / steps if steps > 0 else 0)
            
            # Calculate rolling statistics
            window_size = 100
            current_avg_score = np.mean(scores[-window_size:]) if len(scores) >= window_size else np.mean(scores)
            avg_scores.append(current_avg_score)
            
            # Update learning rate based on performance
            agent.scheduler.step(current_avg_score)
            
            # Save best model
            if current_avg_score > best_avg_score:
                best_avg_score = current_avg_score
                metrics = {
                    'episode': episode,
                    'avg_score': current_avg_score,
                    'best_score': max(scores),
                    'window_size': window_size
                }
                agent.save_model(f"{save_dir}/best_model.pth", metrics)
                print(f"\nNew best model saved! Average score: {current_avg_score:.2f}")
            
            # Regular checkpoints
            if episode % SAVE_INTERVAL == 0:
                agent.save_model(f"{save_dir}/checkpoint_{episode}.pth")
            
            # Update history
            history['scores'].append(env.score)
            history['avg_scores'].append(current_avg_score)
            history['losses'].append(losses[-1])
            history['steps'].append(steps)
            history['epsilon'].append(agent.epsilon)
            
            # Save training history
            with open(f"{save_dir}/history.json", 'w') as f:
                json.dump(history, f)
            
            # Print and plot progress
            if (episode + 1) % PRINT_INTERVAL == 0:
                print(f"Episode: {episode+1}")
                print(f"Score: {env.score}")
                print(f"Average Score (last {window_size}): {current_avg_score:.2f}")
                print(f"Steps: {steps}")
                print(f"Epsilon: {agent.epsilon:.3f}")
                print(f"Loss: {losses[-1]:.5f}")
                print("-" * 50)
                
                # Update plots
                figure.clf()
                
                # Score plot
                plt.subplot(311)
                plt.plot(scores, 'b.', alpha=0.3, label='Score')
                plt.plot(avg_scores, 'r-', label=f'{window_size}-ep average')
                plt.axhline(y=best_avg_score, color='g', linestyle='--', label=f'Best Avg: {best_avg_score:.2f}')
                plt.title(f'Training Progress (Episode {episode+1})')
                plt.ylabel('Pipes Passed')
                plt.grid(True)
                plt.legend()

                # Steps plot
                plt.subplot(312)
                plt.plot(steps_per_episode, 'c.', alpha=0.3, label='Steps')
                plt.plot(np.convolve(steps_per_episode, np.ones(window_size)/window_size, mode='valid'),
                        'r-', label=f'{window_size}-ep average')
                plt.ylabel('Steps per Episode')
                plt.grid(True)
                plt.legend()

                # Loss and Epsilon plot
                plt.subplot(313)
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                
                # Plot loss
                ax1.plot(losses, 'm.', alpha=0.3, label='Loss')
                ax1.plot(np.convolve(losses, np.ones(window_size)/window_size, mode='valid'),
                        'r-', label=f'{window_size}-ep average')
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Loss', color='m')
                ax1.tick_params(axis='y', labelcolor='m')
                ax1.grid(True)
                
                # Plot epsilon on same subplot with different y-axis
                epsilon_line = ax2.plot(history['epsilon'], 'y-', label='Epsilon', alpha=0.5)
                ax2.set_ylabel('Epsilon', color='y')
                ax2.tick_params(axis='y', labelcolor='y')
                
                # Combine legends
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

                # Adjust layout and display
                plt.tight_layout()
                figure.canvas.draw()
                figure.canvas.flush_events()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
    finally:
        print("\nSaving final model and cleaning up...")
        # Save final model state
        if len(scores) >= window_size:
            final_avg_score = np.mean(scores[-window_size:])
            torch.save({
                'episode': episode,
                'model_state_dict': agent.policy_net.state_dict(),
                'target_state_dict': agent.target_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'score': final_avg_score,
                'window_size': window_size,
                'epsilon': agent.epsilon,
                'is_final': True
            }, f"{save_dir}/final_model.pth")
            print(f"Final model saved with average score: {final_avg_score:.2f}")

        # Clean up
        env.close()
        pygame.quit()

if __name__ == "__main__":
    from spil_opdatering import FlappyBird  # Import your FlappyBird environment
    from agent_opdateret import DQNAgent           # Import your DQNAgent class
    
    train_flappy(env_class=FlappyBird, agent_class=DQNAgent)