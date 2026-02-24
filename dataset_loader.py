import torch
from torch.utils.data import Dataset
import chess
import numpy as np
from data_processing import board_to_tensor

class ChessDataset(Dataset):
    """
    A PyTorch Dataset that loads FEN strings and their evaluations.
    This allows PyTorch to efficiently load data in parallel batches 
    without crushing the computer's RAM.
    """
    def __init__(self, fen_list, evals_list):
        self.fens = fen_list
        # Normalize the evaluations between -1 and 1
        # E.g., if a position is +1000 centipawns, we cap it or scale it.
        # Let's say +1000 centipawns represents a forced win (1.0).
        self.evals = np.clip(np.array(evals_list) / 1000.0, -1.0, 1.0)
        
    def __len__(self):
        return len(self.fens)
        
    def __getitem__(self, idx):
        # 1. Get the FEN string for this specific index
        fen = self.fens[idx]
        
        # 2. Convert FEN string to a python-chess Board
        try:
            board = chess.Board(fen)
        except ValueError:
            # If the FEN is somehow invalid, just return an empty board
            board = chess.Board()
            
        # 3. Convert the Board into our math Tensor (8x8x12)
        tensor = board_to_tensor(board)
        
        # 4. PyTorch expects (Channels, Height, Width)
        tensor = np.transpose(tensor, (2, 0, 1))
        
        # 5. Get the matching true evaluation score
        target = np.array([self.evals[idx]], dtype=np.float32)
        
        # 6. Return them both as PyTorch Tensors
        return torch.tensor(tensor, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
