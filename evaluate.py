import chess

# Piece values are usually tracked in "centipawns"
# 100 centipawns = 1 Pawn
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0 # The King isn't evaluated by material because it can't be captured!
}

def evaluate_board(board):
    """
    Evaluates the current board state based purely on material (pieces).
    Positive value means White is winning.
    Negative value means Black is winning.
    0 means the material is perfectly even.
    """
    
    # 1. Check for Checkmate first!
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return -99999 # Black delivered checkmate, so Black is winning by a lot!
        else:
            return 99999  # White delivered checkmate, White wins!
    
    # 2. Check for Draw (Stalemate, insufficient material, etc.)
    if board.is_game_over():
        return 0 
    
    # 3. Count the material
    evaluation = 0
    for square in chess.SQUARES: # Loop through all 64 squares
        piece = board.piece_at(square)
        if piece is None:
            continue
            
        value = PIECE_VALUES[piece.piece_type]
        
        # Add score if White piece, subtract score if Black piece
        if piece.color == chess.WHITE:
            evaluation += value
        else:
            evaluation -= value
            
    return evaluation

def main():
    board = chess.Board()
    print("--- Starting Position ---")
    print(f"Evaluation: {evaluate_board(board)} centipawns\n")
    
    # Let's simulate a blunder! We manually remove White's Queen from the board.
    board.remove_piece_at(chess.D1)
    
    print("--- Oh no, White dropped the Queen! ---")
    print(board)
    print(f"Evaluation: {evaluate_board(board)} centipawns (Black is up 900 points!)\n")
    
    # Let's say White takes Black's Rook to compensate
    board.remove_piece_at(chess.A8)
    
    print("--- White takes a Black Rook in return ---")
    print(f"Evaluation: {evaluate_board(board)} centipawns (Black is still up 400 points!)")

if __name__ == "__main__":
    main()
