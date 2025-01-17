# import pygame
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from simpel_spil import FlappyBird

# # Modified DQN architecture to match the trained model
# class DQN(nn.Module):
#     def __init__(self, state_size, action_size, hidden_size):
#         super(DQN, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(state_size, hidden_size),    # layer 0
#             nn.ReLU(),                            # layer 1
#             nn.Linear(hidden_size, hidden_size),   # layer 2
#             nn.ReLU(),                            # layer 3
#             nn.Linear(hidden_size, hidden_size),   # layer 4
#             nn.ReLU(),                            # layer 5
#             nn.Linear(hidden_size, action_size)    # layer 6
#         )
    
#     def forward(self, x):
#         return self.network(x)

# def play_trained_model(model_path='flappy_best_model.pth', num_games=5):
#     """
#     Load and play the trained Flappy Bird model
    
#     Args:
#         model_path (str): Path to the saved model
#         num_games (int): Number of games to play
#     """
#     # Initialize environment
#     env = FlappyBird()
    
#     # Initialize the DQN with the same architecture as during training
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = DQN(state_size=5, action_size=2, hidden_size=128).to(device)
    
#     try:
#         # Load the saved model
#         checkpoint = torch.load(model_path, map_location=device)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         print(f"Loaded model from episode {checkpoint['episode']} with score {checkpoint['score']}")
        
#         # Set evaluation mode
#         model.eval()
        
#         # Play multiple games
#         for game in range(num_games):
#             state = env.reset()
#             done = False
#             game_score = 0
            
#             while not done:
#                 # Convert state to tensor
#                 state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                
#                 # Get action from model
#                 with torch.no_grad():
#                     q_values = model(state_tensor)
#                     action = q_values.max(1)[1].item()
                
#                 # Take action in environment
#                 next_state, reward, done = env.step(action)
#                 state = next_state
#                 game_score = env.score
                
#                 # Render the game
#                 env.render()
#                 pygame.event.pump()
                
#                 # Add a small delay to make the game visible
#                 pygame.time.wait(20)
            
#             print(f"Game {game + 1} Score: {game_score}")
            
#     except FileNotFoundError:
#         print(f"Error: Could not find model file at {model_path}")
#     except Exception as e:
#         print(f"Error loading or running model: {str(e)}")
#         raise  # Re-raise the exception to see the full traceback
#     finally:
#         env.close()
#         pygame.quit()

# if __name__ == "__main__":
#     play_trained_model()


import torch

# Load the checkpoint
checkpoint = torch.load('flappy_best_model.pth')

# Print the information
print(f"Model saved from episode {checkpoint['episode']} with score {checkpoint['score']}")