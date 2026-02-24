import torch
import torch.nn as nn
import torch.optim as optim
from network import ChessNet

def main():
    print("--- Setting up Training ---")
    
    # 1. Load the Brain
    model = ChessNet()
    
    # 2. The Loss Function (How wrong was the AI?)
    # Mean Squared Error (MSE) is great for giving higher penalties to bigger mistakes.
    criterion = nn.MSELoss()
    
    # 3. The Optimizer (How the AI adjusts its brain to be less wrong)
    # Adam is the most popular optimizer. It slowly nudges the internal weights.
    # lr = Learning Rate (how big of a step it takes to fix its mistake)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Let's create some FAKE data to pretend we are training it on real games:
    # 32 random chess positions (Batch Size = 32)
    fake_board_tensors = torch.rand((32, 12, 8, 8)) 
    
    # 32 fake true evaluations (e.g. Grandmaster says position 1 is +0.5, position 2 is -0.9)
    # We make them random numbers between -1 and 1
    true_evaluations = torch.rand((32, 1)) * 2 - 1
    
    print("Beginning Training Loop for 10 Epochs (Passes over the data)...")
    
    # 5. The Training Loop!
    for epoch in range(1, 11):
        # Step 1: Tell the AI to evaluate the 32 boards using its current (random) brain
        predictions = model(fake_board_tensors)
        
        # Step 2: Compare the AI's predictions to what the Grandmaster said it should be
        loss = criterion(predictions, true_evaluations)
        
        # Step 3: Zero out the math gradients from the last step (PyTorch requirement)
        optimizer.zero_grad()
        
        # Step 4: BACKPROPAGATION - The magic of Deep Learning. 
        # PyTorch calculates exactly *which* weights in the CNN caused the mistake.
        loss.backward()
        
        # Step 5: Nudge the weights to be slightly more accurate next time
        optimizer.step()
        
        print(f"Epoch {epoch}/10 | Loss/Mistake amount: {loss.item():.4f}")

    print("\nTraining Complete! The mistake amount (Loss) should have gone down!")
    print("(Note: It learned absolutely nothing useful because we trained it on random noise, but the math works!)")

if __name__ == "__main__":
    main()
