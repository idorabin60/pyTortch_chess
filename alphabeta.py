import chess
from evaluate import evaluate_board

def quiescence_search(board, alpha, beta, is_maximizing, eval_func):
    stand_pat = eval_func(board)

    if is_maximizing:
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
        
        capture_moves = [m for m in board.legal_moves if board.is_capture(m)]
        for move in capture_moves:
            board.push(move)
            score = quiescence_search(board, alpha, beta, False, eval_func)
            board.pop()
            
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha
    else:
        if stand_pat <= alpha:
            return alpha
        beta = min(beta, stand_pat)
        
        capture_moves = [m for m in board.legal_moves if board.is_capture(m)]
        for move in capture_moves:
            board.push(move)
            score = quiescence_search(board, alpha, beta, True, eval_func)
            board.pop()
            
            if score <= alpha:
                return alpha
            beta = min(beta, score)
        return beta


def minimax_alpha_beta(board, depth, alpha, beta, is_maximizing, eval_func=evaluate_board):
    """
    Minimax with Alpha-Beta Pruning.
    alpha: The best score White can guarantee (initially -infinity)
    beta: The best score Black can guarantee (initially +infinity)
    """
    
    if depth == 0 or board.is_game_over():
        return quiescence_search(board, alpha, beta, is_maximizing, eval_func)

    if is_maximizing:
        best_eval = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_score = minimax_alpha_beta(board, depth - 1, alpha, beta, False, eval_func)
            board.pop()
            
            best_eval = max(best_eval, eval_score)
            alpha = max(alpha, eval_score) # White updates its guaranteed minimum score
            
            # THE PRUNING STEP:
            # If White found a move that is ALREADY better than the best score Black can force (beta),
            # Black will NEVER let White play this line.
            # So, we stop looking at any more moves in this branch!
            if beta <= alpha:
                break # "Prune" the tree ✂️
                
        return best_eval

    else:
        # Black's Turn
        best_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_score = minimax_alpha_beta(board, depth - 1, alpha, beta, True, eval_func)
            board.pop()
            
            best_eval = min(best_eval, eval_score)
            beta = min(beta, eval_score) # Black updates its guaranteed maximum score
            
            # THE PRUNING STEP:
            # If Black found a move that is ALREADY worse (for White) than the best score White can force (alpha),
            # White will NEVER let Black play this line.
            # So, we stop looking!
            if beta <= alpha:
                break # "Prune" the tree ✂️
                
        return best_eval

def get_best_move_alpha_beta(board, depth):
    alpha = -float('inf')
    beta = float('inf')
    best_move = None
    
    if board.turn == chess.WHITE:
        best_eval = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_score = minimax_alpha_beta(board, depth - 1, alpha, beta, False)
            board.pop()
            
            if eval_score > best_eval:
                best_eval = eval_score
                best_move = move
            # Alpha is updated at the root as well
            alpha = max(alpha, eval_score) 
    else:
        best_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_score = minimax_alpha_beta(board, depth - 1, alpha, beta, True)
            board.pop()
            
            if eval_score < best_eval:
                best_eval = eval_score
                best_move = move
            # Beta is updated at the root as well
            beta = min(beta, eval_score)
                
    return best_move

def main():
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 3")
    print("--- Current Position ---")
    print(board)
    print("\nLooking for best move (Depth 3) using Alpha-Beta Pruning...")
    
    # Notice we can cleanly look at Depth 3 now much faster!
    best_move = get_best_move_alpha_beta(board, depth=3)
    print(f"\nThe engine chose: {best_move}")

if __name__ == "__main__":
    main()
