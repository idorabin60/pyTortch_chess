import chess

def main():
    board = chess.Board()

    print("--- Starting Position ---")
    print(board)
    print("\n")

    # 1. Pushing a move
    # Let's say White plays e4. We create a Move object using UCI format.
    move_e4 = chess.Move.from_uci("e2e4")
    
    # We "push" the move onto the board's internal stack
    board.push(move_e4)

    print("--- After White plays e4 ---")
    print(board)
    print(f"Is it White's turn? {board.turn == chess.WHITE}\n")

    # 2. Pushing another move
    # Now Black plays e5
    move_e5 = chess.Move.from_uci("e7e5")
    board.push(move_e5)

    print("--- After Black plays e5 ---")
    print(board)
    print("\n")

    # 3. Unmaking a move (Popping)
    # The AI needs to test moves and then take them back to test other moves!
    # "Pop" removes the very last move played.
    undone_move = board.pop()
    
    print(f"--- After taking back {undone_move.uci()} ---")
    print(board)

if __name__ == "__main__":
    main()
