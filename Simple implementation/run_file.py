import pygame
import torch
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from simpel_spil import FlappyBird
from agent import DQNAgent
#importerer nødevendige moduler og klasser

# Simple replay buffer 
class ReplayBuffer:
    #class der lagrer tidligere spiloplevelser
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        #der sættes en øvre grænse for hvor mange spilgentagelser der kan gemmes (capacity)
        #deque funktionen af max capacity gør, at når capacity er fyldt, at det ældste info ryger og erstattes
            #deque står for double-ended queue

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        #tilføjer en ny spiloplevelse, som en tuple, til bufferen

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        #trækker tilfældig batch fra bufferen, hvis bufferen er for lille, tages så mange elementer som muligt
        states, actions, rewards, next_states, dones = zip(*batch)
        #splitter spiloplevelsen op i 5 arrays 
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
        #konverterer til numpy for effektivisering

    def __len__(self):
        return len(self.buffer)
        #returnerer antal gemte elementer i bufferen

# Training settings
EPISODES = 10000
BATCH_SIZE = 64
BUFFER_CAPACITY = 10000
PRINT_INTERVAL = 50  # How often to print and plot

# Initialize everything
env = FlappyBird()      #opretter spillets miljø ved at hente 
agent = DQNAgent(state_size=5, action_size=2, hidden_size=64)  # Fixed state_size to 5
replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

# Lists for tracking metrics
scores = []  # List to track scores
losses = []  # List to track losses
steps_per_episode = []  # List to track steps per episode
best_score = float('-inf')  # Keep track of the best score

# Training loop
try:
    for episode in range(EPISODES): # kører igennem det antal givne episoder
        state = env.reset()         # nulstiller miljø og returnerer til starttilstanden
        episode_reward = 0          # samler rewards for den enkelte spil-episode
        episode_loss = 0            # samler loss for den enkelte spilepisode
        gradient_steps = 0          # antal gradientopdateringer i denne periode
        #hvorfor er denne nul?
        done = False                # indikation om episode er færdig eller ej
        steps = 0                   # tæller antal skridt i episoden
        
        while not done: #loop der kører indil episoden er færdig/afsluttes
            # Get action from agent
            action = agent.act(state)
            
            # Take action in environment 
            next_state, reward, done = env.step(action)
            
            # Store transition in replay buffer (hvad for en transition? den der lige er spillet?)
            replay_buffer.push(state, action, reward, next_state, done)
            
            # Train if enough samples in replay buffer
            if len(replay_buffer) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
                loss = agent.train_on_batch(states, actions, rewards, next_states, dones)
                episode_loss += loss #samler tab fra træningen
                gradient_steps += 1 #øger tælleren for gradienopdateringer - hvad betyder det i praksis? 
            
            # Update state and metrics
            state = next_state
            episode_reward += reward  # Tilføj den modtagne belønning til episodens samlede belønning
            steps += 1  # Øger tælleren for skridt i episoden
            
            # Render (=gengive) every n'th episode
            #for at se progression
            if episode % 50 == 0:
                env.render()    # viser det seneste spil af de 50. (dvs. der spilles 49 skjulte spil og det sidste vises så)
                pygame.event.pump()  #gør således, at man som bruger kan lukke spillet, uden at det "fryser fast"
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt   # Stopper træningen, hvis brugeren afslutter
        
        # Update target network periodically
        if episode % agent.target_update == 0:
            agent.update_target_network()
        
        # Store episode metrics
        scores.append(episode_reward)   #gemmer totale belønning for episoden
        steps_per_episode.append(steps)   #gemmer antal skridt i episoden
        losses.append(episode_loss / (gradient_steps + 1) if gradient_steps > 0 else 0)  #gennemsnitstab for episoden
        
        # Save best model
        if episode_reward > best_score:
            best_score = episode_reward
            torch.save({
                'episode': episode,  # Den aktuelle episode
                'model_state_dict': agent.policy_net.state_dict(),  # Policy-netværkets vægte
                'optimizer_state_dict': agent.optimizer.state_dict(),  # Optimererens tilstand
                'score': best_score,  # Den bedste score hidtil
                'epsilon': agent.epsilon  # Den aktuelle epsilonværdi (udforskningsrate)
            }, 'flappy_best_model.pth')  # Gemmer modellen i en fil
        
        # Print and plot progress every PRINT_INTERVAL episodes
        if (episode + 1) % PRINT_INTERVAL == 0:
            # Calculate average score over the last PRINT_INTERVAL episodes
            avg_score = np.mean(scores[-PRINT_INTERVAL:]) if len(scores) >= PRINT_INTERVAL else np.mean(scores)
            avg_steps = np.mean(steps_per_episode[-PRINT_INTERVAL:]) if len(steps_per_episode) >= PRINT_INTERVAL else np.mean(steps_per_episode)
            print(f"Episode: {episode+1}, Avg Score: {avg_score:.2f}, Avg Steps: {avg_steps:.1f}, Epsilon: {agent.epsilon:.3f}")
            
            # Plot metrics (opdaterer grafer for resultater)
            plt.figure(1)
            plt.clf()  # Clear the previous plot
            plt.subplot(311)   #plot for scoren
            plt.plot(scores, '.')
            plt.title(f'Training Progress (ε={agent.epsilon:.3f})')
            plt.ylabel('Score')
            plt.grid(True)
            
            #plot for antal skridt pr. episode
            plt.subplot(312)  
            plt.plot(steps_per_episode, '.')
            plt.ylabel('Steps')
            plt.grid(True)
            
            # Plot for tabet (loss)
            plt.subplot(313)
            plt.plot(losses, '.')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.grid(True)
            
            plt.pause(0.1)  # Pause to allow the plot to update

except KeyboardInterrupt:
    print("\nTraining interrupted by user")

#lukker miljøet:
finally:
    env.close()  # Close the environment and quit pygame
    pygame.quit()


