import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        
        # Input channels: 12 (the 8x8x12 board representation)
        # We use Convolutional Neural Networks (CNNs) because they are great at
        # finding spatial patterns (like pieces protecting each other on a grid!)
        
        # 1st Convolutional Layer: Sweeps across the 8x8 board looking for basic patterns
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=64, kernel_size=3, padding=1)
        
        # 2nd Convolutional Layer: Looks for more complex patterns
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Fully Connected Layers: Takes the patterns and turns them into a single evaluation score
        # 128 channels * 8 width * 8 height = 8192 numbers coming out of the convolutional layers
        self.fc1 = nn.Linear(8192, 256)
        self.fc2 = nn.Linear(256, 1) # Output is just ONE number: the evaluation score!
        
    def forward(self, x):
        # Pass input through conv1, then apply ReLU activation function (adds non-linearity)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten the 3D tensor into a 1D line of numbers for the Fully Connected layers
        x = x.view(-1, 8192)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        
        # We use tanh to squash the final output to always be between -1 (Black winning) and 1 (White winning)
        return torch.tanh(x)

def main():
    print("Initializing our Neural Network...")
    model = ChessNet()
    print("\nModel Architecture:")
    print(model)
    
    print("\n--- Testing with a Blank Board ---")
    # PyTorch expects the shape: (Batch Size, Channels, Height, Width)
    # We pass in 1 board, 12 channels, 8 rows, 8 columns of all zeros
    dummy_input = torch.zeros((1, 12, 8, 8))
    
    # Let the untrained model evaluate the blank board
    # It will use completely random starting weights!
    evaluation = model(dummy_input)
    print(f"Untrained Model Evaluation: {evaluation.item():.4f}")
    print("Notice how the score isn't perfect 0! It's guessing randomly because it hasn't learned anything yet.")

if __name__ == "__main__":
    main()
