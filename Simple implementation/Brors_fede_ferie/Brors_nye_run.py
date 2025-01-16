import pygame
import torch
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from simpel_spil_Bror import FlappyBird
from brors_agent import DQNAgent

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

def train_flappy():
    # Training settings
    EPISODES = 50000
    BATCH_SIZE = 256
    BUFFER_CAPACITY = 100000
    PRINT_INTERVAL = 50
    SAVE_WINDOW = 100

    # Initialize everything
    env = FlappyBird()
    agent = DQNAgent(state_size=5, action_size=2, hidden_size=128)
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

    # Lists for tracking metrics
    scores = []
    losses = []
    steps_per_episode = []
    best_avg_score = float('-inf')

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
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                replay_buffer.push(state, action, reward, next_state, done)
                
                if len(replay_buffer) >= BATCH_SIZE:
                    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
                    loss = agent.train_on_batch(states, actions, rewards, next_states, dones)
                    episode_loss += loss
                    gradient_steps += 1
                
                state = next_state
                episode_reward += reward
                steps += 1

                if episode % 50 == 0:
                    env.render()
                    pygame.event.pump()

            # Track metrics
            scores.append(env.score)
            steps_per_episode.append(steps)
            losses.append(episode_loss / (gradient_steps + 1) if gradient_steps > 0 else 0)
            
            # Save model based on consistent performance
            if (episode + 1) % PRINT_INTERVAL == 0 and len(scores) >= SAVE_WINDOW:
                current_avg_score = np.mean(scores[-SAVE_WINDOW:])
                
                if current_avg_score > best_avg_score:
                    best_avg_score = current_avg_score
                    torch.save({
                        'episode': episode,
                        'model_state_dict': agent.policy_net.state_dict(),
                        'target_state_dict': agent.target_net.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                        'score': current_avg_score,
                        'window_size': SAVE_WINDOW,
                        'epsilon': agent.epsilon
                    }, 'flappy_best_model.pth')
                    print(f"\nNew best model saved! Average score over {SAVE_WINDOW} episodes: {current_avg_score:.2f}")
            
            # Print and plot progress
            if (episode + 1) % PRINT_INTERVAL == 0:
                avg_score = np.mean(scores[-PRINT_INTERVAL:])
                avg_steps = np.mean(steps_per_episode[-PRINT_INTERVAL:])
                print(f"Episode: {episode+1}, Avg Score (Pipes): {avg_score:.2f}, Avg Steps: {avg_steps:.1f}, Epsilon: {agent.epsilon:.3f}")
                
                plt.figure(1)
                plt.clf()
                
                plt.subplot(311)
                plt.plot(scores, '.')
                plt.title(f'Training Progress (ε={agent.epsilon:.3f})')
                plt.ylabel('Pipes Passed')
                plt.grid(True)
                
                if len(scores) >= SAVE_WINDOW:
                    rolling_avg = [np.mean(scores[max(0, i-SAVE_WINDOW):i]) for i in range(SAVE_WINDOW, len(scores))]
                    plt.plot(range(SAVE_WINDOW, len(scores)), rolling_avg, 'r-', alpha=0.5, label='100-ep average')
                    plt.axhline(y=best_avg_score, color='g', linestyle='--', label=f'Best Avg: {best_avg_score:.2f}')
                    plt.legend()
                
                plt.subplot(312)
                plt.plot(steps_per_episode, '.')
                plt.ylabel('Steps')
                plt.grid(True)
                
                plt.subplot(313)
                plt.plot(losses, '.')
                plt.xlabel('Episode')
                plt.ylabel('Loss')
                plt.grid(True)
                
                plt.pause(0.1)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    finally:
        if len(scores) >= SAVE_WINDOW:
            final_avg_score = np.mean(scores[-SAVE_WINDOW:])
            torch.save({
                'episode': episode,
                'model_state_dict': agent.policy_net.state_dict(),
                'target_state_dict': agent.target_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'score': final_avg_score,
                'window_size': SAVE_WINDOW,
                'epsilon': agent.epsilon,
                'is_final': True
            }, 'flappy_final_model.pth')
            print(f"\nFinal model saved with average score: {final_avg_score:.2f}")

        env.close()
        pygame.quit()

if __name__ == "__main__":
    train_flappy()