import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        # Simplified architecture with one hidden layer for faster computation
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Use faster ReLU implementation
        self.activation = nn.ReLU(inplace=True)
        
        # Initialize weights with simpler uniform distribution
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc2.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        # Simplified forward pass without batch normalization
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.activation(self.fc1(x))
        return self.fc2(x)

class DQNAgent:
    def __init__(self, state_size=5, action_size=2, hidden_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Smaller hidden size for faster computation
        self.policy_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimized hyperparameters
        self.batch_size = 64  # Smaller batch size for faster training
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        
        # Use SGD with momentum instead of Adam for faster updates
        self.optimizer = optim.SGD(
            self.policy_net.parameters(),
            lr=self.learning_rate,
            momentum=0.9
        )
        
        # Training counter
        self.training_steps = 0
        
    @torch.no_grad()  # Decorator for faster inference
    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).to(self.device)
        q_values = self.policy_net(state)
        return q_values.argmax().item()
    
    def train_on_batch(self, states, actions, rewards, next_states, dones):
        # Convert to tensors efficiently
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q-values and target Q-values efficiently
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and optimize
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon less frequently
        if self.training_steps % 4 == 0:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        self.training_steps += 1
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, filepath):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps
        }, filepath)
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']