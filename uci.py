import sys
import chess
from integration import get_best_move_with_ai

def main():
    """
    The main UCI loop.
    This reads commands from standard input (what the GUI sends us)
    and writes responses to standard output (what we send back to the GUI).
    """
    board = chess.Board()
    
    # We loop forever, listening for commands from the GUI
    while True:
        try:
            line = sys.stdin.readline().strip()
        except EOFError:
            break
            
        if not line:
            continue
            
        # 1. 'uci' command: The GUI wants to know who we are
        if line == "uci":
            sys.stdout.write("id name Antigravity Chess AI\n")
            sys.stdout.write("id author Ido\n")
            sys.stdout.write("uciok\n") # This tells the GUI we are ready!
            sys.stdout.flush()
            
        # 2. 'isready' command: The GUI is checking if we are frozen
        elif line == "isready":
            sys.stdout.write("readyok\n")
            sys.stdout.flush()
            
        # 3. 'ucinewgame' command: The GUI is starting a new game
        elif line == "ucinewgame":
            board = chess.Board()
            
        # 4. 'position' command: The GUI is telling us what is on the board
        # e.g.: "position startpos moves e2e4 e7e5"
        elif line.startswith("position"):
            # Split the command into parts
            parts = line.split()
            
            # Reset the board whether it's 'startpos' or 'fen'
            if "startpos" in parts:
                board = chess.Board()
            elif "fen" in parts:
                try:
                    fen_index = parts.index("fen")
                    # The FEN string is usually everything after 'fen' until 'moves'
                    # But for simplicity, let's assume standard FEN here
                    fen_string = " ".join(parts[fen_index+1:fen_index+7])
                    board = chess.Board(fen_string)
                except Exception as e:
                    # Ignore malformed FEN for now
                    pass
            
            # Now, apply all the moves that have been played in the game so far
            if "moves" in parts:
                moves_index = parts.index("moves")
                for move_str in parts[moves_index + 1:]:
                    board.push_uci(move_str)
                    
        # 5. 'go' command: The GUI wants us to think and make a move!
        # e.g.: "go wtime 300000 btime 300000" or "go depth 3"
        elif line.startswith("go"):
            # For this MVP, we will just ignore exactly what the GUI is asking for 
            # (time limits, specific depths) and always search at Depth 2.
            # Depth 2 ensures the untrained AI replies instantly.
            
            # (In production, you'd parse 'wtime', 'btime', 'depth' and pass them
            # to a much smarter time-management function)
            depth_to_search = 2
            
            best_move = get_best_move_with_ai(board, depth=depth_to_search)
            
            # If the AI couldn't find a move (e.g. checkmate), just return a null move
            if best_move is None:
                sys.stdout.write("bestmove 0000\n")
            else:
                # The crucial final step: tell the GUI our move!
                sys.stdout.write(f"bestmove {best_move.uci()}\n")
            
            sys.stdout.flush()
            
        # 6. 'quit' command: The GUI is closing
        elif line == "quit":
            break

if __name__ == "__main__":
    main()
