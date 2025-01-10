import torch
import numpy as np
from collections import deque
import random
import pickle
import os


class ReplayBuffer:
    def __init__(self, capacity, save_dir='replay_buffer'):
        """
        Initialize Replay Buffer
        Args:
            capacity: Maximum number of experiences to store
            save_dir: Directory to save buffer contents
        """
        self.buffer = deque(maxlen=capacity)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def push(self, state, action, reward, next_state, done):
        """Save an experience to the buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        # Convert to torch tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        return states, actions, rewards, next_states, dones
    
    def save(self, episode):
        """Save buffer to disk"""
        filename = os.path.join(self.save_dir, f'buffer_episode_{episode}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(list(self.buffer), f)
        print(f"Saved replay buffer to {filename}")
    
    def load(self, episode):
        """Load buffer from disk"""
        filename = os.path.join(self.save_dir, f'buffer_episode_{episode}.pkl')
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.buffer = deque(pickle.load(f), maxlen=self.buffer.maxlen)
            print(f"Loaded replay buffer from {filename}")
            return True
        return False
    
    def __len__(self):
        return len(self.buffer)