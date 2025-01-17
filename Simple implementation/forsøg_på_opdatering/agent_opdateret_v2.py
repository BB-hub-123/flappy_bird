import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        # Just 3 layers but with good size ratios
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
        # Layer normalization for stability
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size // 2)
        
        # Small dropout
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights carefully
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.uniform_(layer.weight, -0.1, 0.1)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size=5, action_size=2, hidden_size=256):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.policy_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Tuned hyperparameters
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9997  # Slightly slower decay
        self.learning_rate = 0.0001
        self.tau = 0.001
        
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        self.training_steps = 0
    
    def soft_update(self):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
    
    @torch.no_grad()
    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).to(self.device)
        q_values = self.policy_net(state)
        return q_values.argmax().item()
    
    def train_on_batch(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Double Q-learning
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Huber loss for stability
        loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update target network
        self.soft_update()
        
        # Decay epsilon
        if self.training_steps % 4 == 0:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        self.training_steps += 1
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, filepath):
        torch.save(self.policy_net.state_dict(), filepath)
    
    def load_model(self, filepath):
        self.policy_net.load_state_dict(torch.load(filepath))
        self.target_net.load_state_dict(self.policy_net.state_dict())