import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
#importerer nødvendige moduler

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        #input size = antal inputneuroner - svarer til spillets tilstand 
            #altså for os er det de 7 parametre om fuglens tilstand (højde, afstand, tyngdekraft, rørene, m.m.)
        #hidden size = antal neuroner i skjulte lag
        #output size = antal outputneuroner - svarer til mulige actions
            #altså for os 2 (flap/ikke flap)
            
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),     # Første fuldt forbundne lag (input → skjult).
            nn.ReLU(),                              # ReLU-aktiveringsfunktion.
            nn.Linear(hidden_size, hidden_size),    # Andet fuldt forbundne lag (skjult → skjult).
            nn.ReLU(),                              # ReLU-aktiveringsfunktion.
            nn.Linear(hidden_size, output_size)     # Sidste lag (skjult → output).
        )
        #består af tre lineære lag, hver har vægte og biases
        # og to ReLU-aktiveringsfunktioner, som lærer ikke lineære sammenhænge
        # outputter er så en liste af Q-værdier - et element for hver handling, der kan tages - fx [0.15, 0.78] = ikke flap
    
    def forward(self, x):   #fremdadpropagering 
        return self.network(x)
        #tager indput x (= repræsenterer en tilstand, altså de 7 parametre), sender den gennem netværket, for at beregne Q-værdierne
    

class DQNAgent:
    def __init__(self, state_size=5, action_size=2, hidden_size=64, learning_rate=0.001):  # Added learning_rate parameter
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #learning rate = hvor store vægt justeringer der skal foretages

        # Q-Networks
        self.policy_net = DQN(state_size, hidden_size, action_size).to(self.device)   #bruges til at træffe beslutninger
        self.target_net = DQN(state_size, hidden_size, action_size).to(self.device)   #bruges til at beregne Q-værdier under træning
        self.target_net.load_state_dict(self.policy_net.state_dict())   #synkroniserer taget-netværket med policy-netværkets vægte
        
        # Training parameters
        self.batch_size = 32     #batch-størrelsen = 32 (der udvælges tilfældigt fra replay bufferen)
        self.gamma = 0.99        #
        self.epsilon = 1.0       #udforskningsrate, så hvor ofte den explorer - dette vil falde med tiden
        self.epsilon_min = 0.01  #
        self.epsilon_decay = 0.995
        self.target_update = 10
        self.learning_rate = learning_rate  # Store learning rate as instance variable
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)  # Use learning rate in optimizer
        #bruges til at justere vægte i policy-netværket
        self.criterion = nn.MSELoss()
        #MSE = mean squared loss
        # Bruges som loss-funktion tl at minimere forskel mellem policy_nets forudsagte Q-værdier og de målte værdier fra target_netværket

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        #random decimaltal mellem 0 og 1 (inkl. 0, eksl. 1) - default at det er mellem [0,1[
        
        with torch.no_grad():
            state = np.array(state).flatten()  # Ensure state is flattened
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
        #exploitation - beregner Q-værdien for aktuelle tilstand, og vælger handling med højest Q-værdi baseret på tidligere erfaring  
    
    def train_on_batch(self, states, actions, rewards, next_states, dones):
        #definerer træningsfunktion, der arbejder ud fra batches

        # Convert to numpy arrays first
        states = np.array(states)
        next_states = np.array(next_states)
        
        # Flatten if needed
        if len(states.shape) > 2:
            states = states.reshape(states.shape[0], -1)
            next_states = next_states.reshape(next_states.shape[0], -1)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Get current Q values
        current_q = self.policy_net(states)
        current_q_values = current_q.gather(1, actions.unsqueeze(1)).squeeze()
        #gather = henter Q-værdierne for de handlinger agenten faktisk tog

        # Get target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            #maksimale q-værdier fra target netværket for næste tilstande
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            #beregnes som belønning + fremtidig værdi baseret på gamma og fremtidig værdi
            # done = 0, hvis man er død, = 1, hvis man er i live
                # (1-dones) gør således, at hvis man dør, at kun den umiddelbare belønning tæller 

        # Compute loss and optimize
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # beregner tabet mellem forudagte og målte Q-værdier
        # opdaterer netværkets vægte for at mindske tabet

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        # opdaterer epsilon ved at tage maks-værdien af enten mindste epsilon, som vi har at sat til 0.01, eller epsilon (rekursiv) * epsilon decay

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())