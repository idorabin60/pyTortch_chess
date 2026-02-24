def quiescence_search(board, alpha, beta):
    """
    Quiescence Search (QS)
    This function continues looking *past* the maximum depth limit, but ONLY
    looks at captures (or checks). It's a highly targeted, lightweight minimax.
    """
    # 1. First, we get the "stand pat" score. This is our score if we do nothing
    # and just stop searching right now.
    stand_pat = evaluate_board(board)
    
    # If our score is already better than Beta, our opponent will never let us
    # reach this position anyway. We can prune immediately!
    if stand_pat >= beta:
        return beta
        
    # If our score is better than Alpha, we update Alpha.
    if alpha < stand_pat:
        alpha = stand_pat
        
    # 2. Generate ONLY capturing moves
    # We don't care about quiet positional moves here.
    capture_moves = [move for move in board.legal_moves if board.is_capture(move)]
    
    # Optional: Sort capture moves from most valuable victim to least valuable
    # e.g., Pawn takes Queen is better than Queen takes Pawn!
    
    for move in capture_moves:
        board.push(move)
        # Call QS recursively. Notice how the perspective flips, just like normal Minimax
        score = -quiescence_search(board, -beta, -alpha)
        board.pop()
        
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
            
    return alpha
