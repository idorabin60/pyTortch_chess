import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from network import ChessNet
from dataset_loader import ChessDataset
import wandb # The logging library you requested!

def main():
    print("--- Starting Production Chess Training ---")
    
    # 1. Initialize Weights & Biases for Remote Logging
    # (You will need to run 'wandb login' in your GCP terminal first)
    wandb.init(project="chess-antigravity", name="stockfish-evals-run1")
    
    # 2. Check for the NVIDIA L4 GPU!
    # If the VM has CUDA installed properly, PyTorch will see it immediately.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device}")
    
    # 3. Load the model and move it to the GPU
    model = ChessNet().to(device)
    
    print("Loading the dataset (this might take a minute for 16M rows)...")
    import pandas as pd
    
    # Load the actual Kaggle CSV downloaded on the VM
    # The Kaggle file is called 'chessData.csv' and has 'FEN' and 'Evaluation' columns
    df = pd.read_csv("chessData.csv")
    
    # Clean the data: Some evaluations in the dataset are strings like '#+4' (mate in 4).
    # We filter those out to keep pure numerical centipawn scores mapping to standard evaluations.
    df = df[~df['Evaluation'].astype(str).str.contains('#')]
    df['Evaluation'] = df['Evaluation'].astype(float)
    
    real_fens = df['FEN'].tolist()
    real_evals = df['Evaluation'].tolist()
    
    dataset = ChessDataset(real_fens, real_evals)
    
    # The DataLoader automatically bundles the data into batches of say, 4096 boards
    # and handles tossing them to the GPU asynchronously while the CPU prepares the next batch.
    dataloader = DataLoader(dataset, batch_size=4096, shuffle=True, num_workers=4)
    
    # 5. Training Fundamentals
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Track hyperparams in wandb
    wandb.config = {
      "learning_rate": 0.001,
      "epochs": 100,
      "batch_size": 4096
    }
    
    print("Starting Training Loop!")
    for epoch in range(1, 101):
        model.train() # Set to training mode
        epoch_loss = 0.0
        
        # Loop over every single batch in the 16 million FENs
        for batch_boards, batch_evals in dataloader:
            
            # MOVE the data from CPU Ram specifically onto the GPU VRAM!
            batch_boards = batch_boards.to(device)
            batch_evals = batch_evals.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_boards)
            loss = criterion(predictions, batch_evals)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_boards.size(0)
            
        # Calculate the average mistake amount over the whole epoch
        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch}/100 | Avg Training Loss: {avg_loss:.4f}")
        
        # LOG TO WANDB! 
        # This streams the Loss securely to your web dashboard instantly.
        wandb.log({"epoch": epoch, "loss": avg_loss})
        
        # Save checkpoints safely every 5 epochs
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"antigravity_chess_epoch_{epoch}.pth")
            print(f"Model Checkpoint Saved for Epoch {epoch}!")

    print("Training Finished!")
    # Save the absolute final Golden Model weights
    torch.save(model.state_dict(), "best_model.pth")
    wandb.finish()

if __name__ == "__main__":
    main()
