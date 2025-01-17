import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        # Two hidden layers but still efficient
        self.fc1 = nn.Linear(input_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, output_size)
        
        # Leaky ReLU for better gradient flow
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        
        # Xavier initialization for better initial learning
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size=5, action_size=2, hidden_size=128):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Smaller hidden size for faster computation
        self.policy_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Retuned hyperparameters for better learning
        self.batch_size = 256  # Increased for better stability
        self.gamma = 0.99  # Reduced to focus more on immediate rewards
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # Increased minimum exploration
        self.epsilon_decay = 0.998 # Slower decay for better exploration
        self.learning_rate = 0.003  # Reduced to prevent overshooting
        
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