import torch
import pygame
from simpel_spil_Bror import FlappyBird
from brors_agent import DQNAgent

def load_and_play_model(model_path='flappy_best_model.pth', num_games=5):
    """
    Load and play saved model
    """
    # First check and display model info
    try:
        checkpoint = torch.load(model_path)
        print(f"\nLoaded model information:")
        print(f"Episode: {checkpoint['episode']}")
        print(f"Average Score: {checkpoint['score']:.2f}")
        print(f"Window size used for averaging: {checkpoint.get('window_size', 'Not specified')}")
        
        # Initialize environment and agent
        env = FlappyBird()
        agent = DQNAgent(state_size=5, action_size=2, hidden_size=128)
        
        # Load the saved weights
        agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
        
        # Set to evaluation mode and disable exploration
        agent.policy_net.eval()
        agent.epsilon = 0
        
        # Play games
        total_score = 0
        for game in range(num_games):
            state = env.reset()
            done = False
            while not done:
                # Get action from model
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = agent.policy_net(state_tensor)
                    action = q_values.max(1)[1].item()
                
                # Take action
                next_state, _, done = env.step(action)
                state = next_state
                
                # Render
                env.render()
                pygame.event.pump()
                pygame.time.wait(20)  # Slow down for visualization
            
            game_score = env.score
            total_score += game_score
            print(f"Game {game + 1} Score: {game_score}")
        
        print(f"\nAverage score over {num_games} games: {total_score/num_games:.2f}")
            
    except FileNotFoundError:
        print(f"No saved model found at {model_path}")
    except Exception as e:
        print(f"Error loading or running model: {str(e)}")
        raise
    finally:
        if 'env' in locals():
            env.close()
        pygame.quit()

if __name__ == "__main__":
    load_and_play_model()