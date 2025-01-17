import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        # Network layers - we use a deep architecture with three hidden layers
        # This gives us enough capacity to learn complex game strategies
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        
        # Layer normalization for each hidden layer
        # Unlike BatchNorm, LayerNorm normalizes across features, not batch
        # This makes it perfect for both single samples and batches
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        
        # Initialize weights using He initialization
        # This helps prevent vanishing/exploding gradients with ReLU
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.kaiming_normal_(self.fc4.weight)
        
        # Small positive bias to help with ReLU dead neuron problem
        nn.init.constant_(self.fc1.bias, 0.1)
        nn.init.constant_(self.fc2.bias, 0.1)
        nn.init.constant_(self.fc3.bias, 0.1)
        nn.init.constant_(self.fc4.bias, 0.1)
        
        # Dropout for regularization
        # 0.2 means we keep 80% of neurons active
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # First layer with normalization and activation
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second layer with residual connection
        identity = x
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + identity  # Residual connection helps gradient flow
        
        # Third layer with another residual connection
        identity = x
        x = self.fc3(x)
        x = self.ln3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + identity  # Second residual connection
        
        # Output layer (no normalization or activation)
        x = self.fc4(x)
        return x


class DQNAgent:
    def __init__(self, state_size=5, action_size=2, hidden_size=128, learning_rate=0.0003):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.policy_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Training parameters
        self.batch_size = 64
        self.gamma = 0.99  # Discount factor for future rewards
        self.epsilon = 1.0  # Start with full exploration
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.9995  # Slow decay for better exploration
        self.target_update = 10  # Update target network every N steps
        self.learning_rate = learning_rate
        
        # Initialize optimizer with Adam
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler for adaptive learning
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=1000,
            verbose=True,
            min_lr=0.00001
        )
        
        # Huber loss (smooth L1) for better stability with outliers
        self.criterion = nn.SmoothL1Loss()
        
        # Training step counter
        self.training_steps = 0
    
    def act(self, state, training=True):
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            # Convert state to tensor and ensure proper shape
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            state = state.to(self.device)
                
            # Get Q-values and select best action
            self.policy_net.eval()  # Set to evaluation mode
            q_values = self.policy_net(state)
            self.policy_net.train()  # Set back to training mode
            return q_values.argmax().item()
    
    def train_on_batch(self, states, actions, rewards, next_states, dones):
        # Convert all inputs to appropriate tensor types
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values with Double DQN
        with torch.no_grad():
            # Get actions from policy network
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            # Get Q-values from target network for those actions
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            # Compute target Q values
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update epsilon with decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        self.training_steps += 1
        
        return loss.item()
    
    def update_target_network(self):
        # Soft update of target network
        tau = 0.001  # Soft update parameter
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)
    
    def save_model(self, filepath, metrics=None):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'metrics': metrics
        }, filepath)
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
        return checkpoint.get('metrics', None)