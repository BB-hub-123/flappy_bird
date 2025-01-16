import torch

def check_saved_model(model_path='flappy_best_model.pth'):
    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path)
        
        # Print all saved information
        print(f"\nModel Information from {model_path}:")
        print(f"Saved at episode: {checkpoint['episode']}")
        print(f"Score: {checkpoint['score']:.2f}")
        print(f"Window size used for averaging: {checkpoint.get('window_size', 'Not specified')}")
        print(f"Epsilon value when saved: {checkpoint['epsilon']:.4f}")
        
        # Print network information
        if 'model_state_dict' in checkpoint:
            print("\nModel contains policy network weights")
        if 'target_state_dict' in checkpoint:
            print("Model contains target network weights")
        if 'optimizer_state_dict' in checkpoint:
            print("Model contains optimizer state")
            
    except FileNotFoundError:
        print(f"No saved model found at {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

if __name__ == "__main__":
    # Check both best and final models if they exist
    check_saved_model('flappy_best_model.pth')
    check_saved_model('flappy_final_model.pth')