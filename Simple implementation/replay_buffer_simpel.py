import torch
import numpy as np
from collections import deque
import random
#den her bruges faktisk ikke rigtig så meget hehe
class ReplayBuffer:
    def __init__(self, capacity):
        """
        Initialize Replay Buffer
        Args:
            capacity: Maximum number of experiences to store
        """ 
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Save an experience to the buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        # Unzip the batch into separate arrays
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)