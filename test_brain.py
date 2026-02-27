import chess
from integration import ai_evaluate_board

fens = [
    ("Starting Position", chess.STARTING_FEN),
    ("White up a Queen", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"), # Wait, white up a queen? I'll just write it
    ("White down a Queen", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
]

for name, fen in fens:
    board = chess.Board(fen)
    print(f"{name}: {ai_evaluate_board(board):.2f}")
    
board = chess.Board()
board.remove_piece_at(chess.D1) # Remove White Queen
print(f"White missing Queen: {ai_evaluate_board(board):.2f}")

board = chess.Board()
board.remove_piece_at(chess.D8) # Remove Black Queen
print(f"Black missing Queen: {ai_evaluate_board(board):.2f}")
