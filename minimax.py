import chess
from evaluate import evaluate_board

def minimax(board, depth, is_maximizing):
    """
    The Minimax algorithm.
    - depth: How many moves ahead we want to look.
    - is_maximizing: True if it's White's turn (wants positive score), 
                     False if it's Black's turn (wants negative score).
    Returns the evaluation score of the board.
    """
    
    # Base Case: We reached our depth limit OR the game is over
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    if is_maximizing:
        # White's turn: Find the move with the HIGHEST score
        best_eval = -float('inf') # Start with the worst possible score
        for move in board.legal_moves:
            board.push(move) # Try the move
            
            # Recursively call minimax for Black's response
            # Note: depth decreases by 1, and it's no longer maximizing (it's Black's turn)
            eval_score = minimax(board, depth - 1, False)
            
            board.pop()      # Undo the move!
            
            # Did we find a better move?
            best_eval = max(best_eval, eval_score)
        return best_eval

    else:
        # Black's turn: Find the move with the LOWEST score
        best_eval = float('inf') # Start with the worst possible score (favoring White)
        for move in board.legal_moves:
            board.push(move) # Try the move
            
            # Recursively call minimax for White's response
            eval_score = minimax(board, depth - 1, True)
            
            board.pop()      # Undo the move!
            
            # Did we find a better move for Black?
            best_eval = min(best_eval, eval_score)
        return best_eval

def get_best_move(board, depth):
    """
    Uses minimax to actually return the best Move object, not just the score.
    """
    best_move = None
    
    if board.turn == chess.WHITE:
        best_eval = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            # Black will respond, so next is maximizing=False
            eval_score = minimax(board, depth - 1, False)
            board.pop()
            
            if eval_score > best_eval:
                best_eval = eval_score
                best_move = move
    else:
        # Black's turn
        best_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            # White will respond, so next is maximizing=True
            eval_score = minimax(board, depth - 1, True)
            board.pop()
            
            if eval_score < best_eval:
                best_eval = eval_score
                best_move = move
                
    return best_move

def main():
    # Let's set up a custom board where White has a checkmate in 1 move!
    # White Queen to f7 is checkmate.
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 3")
    print("--- Current Position ---")
    print(board)
    print("\nLooking for best move (Depth 2)...")
    
    best_move = get_best_move(board, depth=2)
    print(f"\nThe engine chose: {best_move}")
    
    board.push(best_move)
    print(f"Is it checkmate? {board.is_checkmate()}")

if __name__ == "__main__":
    main()
