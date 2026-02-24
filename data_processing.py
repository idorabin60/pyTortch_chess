import chess
import numpy as np

def board_to_tensor(board):
    """
    Converts a python-chess board into a 3D NumPy array (tensor)
    Shape: (8, 8, 12)
    - 8 rows
    - 8 columns
    - 12 channels (6 piece types * 2 colors)
    """
    
    # We initialize a tensor of all zeros. 
    # The shape is (8, 8, 12) meaning an 8x8 grid, but with 12 layers!
    tensor = np.zeros((8, 8, 12), dtype=np.float32)

    # Define the mapping from piece type and color to a single channel index (0 to 11)
    # White: Pawns=0, Knights=1, Bishops=2, Rooks=3, Queens=4, Kings=5
    # Black: Pawns=6, Knights=7, Bishops=8, Rooks=9, Queens=10, Kings=11
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # SQUARES go from 0 to 63. We map this to 2D coordinates (row, col)
            row = chess.square_rank(square)
            col = chess.square_file(square)
            
            # Figure out which of the 12 channels this piece belongs to
            channel = piece.piece_type - 1 # piece_type is 1-6, so subtract 1 to get 0-5
            if piece.color == chess.BLACK:
                channel += 6 # Black pieces are channels 6-11
                
            # Put a "1" in the exact spot where the piece is!
            tensor[row][col][channel] = 1.0

    return tensor

def main():
    board = chess.Board()
    print("--- Current Board ---")
    print(board)
    
    print("\n--- Converting to Tensor ---")
    tensor = board_to_tensor(board)
    print(f"Tensor shape (Rows, Cols, Channels): {tensor.shape}")
    
    print("\nLet's look at Channel 0 (White Pawns):")
    # tensor[:, :, 0] gets the entire 8x8 grid for the 1st layer
    print(tensor[:, :, 0])
    
    print("\nNotice the White Pawns are a row of 1s!")
    
    print("\nLet's look at Channel 10 (Black Queen):")
    print(tensor[:, :, 10])

if __name__ == "__main__":
    main()
