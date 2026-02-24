import chess
import torch
from network import ChessNet
from data_processing import board_to_tensor
from alphabeta import minimax_alpha_beta

# 1. Load our trained AI Brain
# Normally, we would load the weights from a saved file: model.load_state_dict(torch.load('best_model.pth'))
# Since we haven't trained on GCP yet, this will just use the random weights!
ai_brain = ChessNet()
ai_brain.eval() # Tell PyTorch we are Evaluating, not Training

def ai_evaluate_board(board):
    """
    Replaces our old material-counting evaluate_board() with our Deep Learning model!
    """
    # 1. Convert the python-chess board into a math tensor
    tensor = board_to_tensor(board)
    
    # 2. PyTorch expects (Batch, Channels, Height, Width).
    # Our data_processing returned (Height, Width, Channels).
    # So we permute it from (8,8,12) -> (12,8,8) and then add the Batch dim.
    tensor = torch.tensor(tensor).permute(2, 0, 1).unsqueeze(0)
    
    # 3. Ask the AI for its opinion!
    with torch.no_grad(): # Tell PyTorch not to track gradients (saves memory/time)
        evaluation = ai_brain(tensor)
        
    # 4. Extract the single number from the tensor (-1 to 1) and scale it 
    # Let's multiply by 1000 so the Minimax algorithm works with centipawns like before!
    return evaluation.item() * 1000

def get_best_move_with_ai(board, depth):
    alpha = -float('inf')
    beta = float('inf')
    best_move = None
    
    # Notice this is the EXACT same Minimax template as Phase 2!
    # The only difference is that deep inside minimax_alpha_beta, 
    # we would call ai_evaluate_board() instead of the old evaluate_board() function!
    
    if board.turn == chess.WHITE:
        best_eval = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            # Pretend minimax_alpha_beta uses ai_evaluate_board internally
            eval_score = -ai_evaluate_board(board) # Just a placeholder call for demonstration
            board.pop()
            
            if eval_score > best_eval:
                best_eval = eval_score
                best_move = move
    else:
        best_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_score = ai_evaluate_board(board)
            board.pop()
            
            if eval_score < best_eval:
                best_eval = eval_score
                best_move = move
                
    return best_move

def main():
    board = chess.Board()
    print("--- Integrating Neural Net with Minimax ---\n")
    
    print(f"Classical Material Evaluation: 0.0")
    print(f"Deep Learning AI Evaluation:   {ai_evaluate_board(board):.4f} centipawns (guessing!)\n")
    
    print("Asking AI to find best move on Starting Board...")
    best_move = get_best_move_with_ai(board, depth=1)
    
    print(f"\nThe Neural Net chose: {best_move}")
    # It will pick a completely random legal move because it's untrained!

if __name__ == "__main__":
    main()
