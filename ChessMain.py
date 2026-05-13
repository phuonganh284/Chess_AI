import tkinter as tk
import ChessEngine
import solver
import chess
import tkinter.messagebox as messagebox

SQ_SIZE = 20
AI_DELAY = 800  # ms
MINIMAX_DEPTH = 2

# white first, black second
# random vs minimax: white -> random moves first, black -> random moves second 
CHESS_TURN = chess.WHITE

PIECE_UNICODE = {
    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟',
}

class ChessUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Game")
        self.game_state = ChessEngine.GameState()

        self.buttons = [[None for _ in range(8)] for _ in range(8)]
        self.selected = None
        self.legal_moves = []
        self.mode = None
        self.create_menu()

    # menu
    def create_menu(self):
        self.clear_screen()
        label = tk.Label(self.root, text="Select Mode", font=("Arial", 20))
        label.pack(pady=20)

        modes = [
            ("1. Player vs Player", "pvp"),
            ("2. Player vs Random AI", "pvr"),
            ("3. Player vs Minimax AI", "pvm"),
            ("4. Random vs Minimax", "rvm"),
            ("5. Minimax vs Minimax", "mvm"),
            ("6. Player vs Machine Learning", "pml"),
            ("7. Random vs Machine Learning", "rml"),
            ("8. Minimax vs Machine Learning", "mml"),
        ]

        for text, mode in modes:
            btn = tk.Button(self.root, text=text,
                            command=lambda m=mode: self.start_game(m),
                            width=40)
            btn.pack(pady=10)

    def start_game(self, mode):
        self.mode = mode
        self.game_state = ChessEngine.GameState()
        self.selected = None
        self.legal_moves = []

        self.clear_screen()
        self.create_board()
        self.update_board()

        self.root.after(AI_DELAY, self.ai_loop)

    def clear_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()


    # board
    def create_board(self):
        self.frame = tk.Frame(self.root)
        self.frame.pack()

        for r in range(8):
            for c in range(8):
                btn = tk.Button(self.frame, width=SQ_SIZE//10, height=(SQ_SIZE-5)//30,
                                command=lambda r=r, c=c: self.on_click(r, c))
                btn.grid(row=r, column=c)
                self.buttons[r][c] = btn

    def update_board(self):
        board = self.game_state.board_to_array()
        for r in range(8):
            for c in range(8):
                piece = board[r][c]
                # convert to unicode
                text = ""
                if piece != "--":
                    color = piece[0]
                    p = piece[1]
                    if p == 'p':
                        symbol = 'P'
                    else:
                        symbol = p

                    if color == 'b':
                        symbol = symbol.lower()
                    text = PIECE_UNICODE[symbol]
                # board color
                bg = "#f0d9b5" if (r + c) % 2 == 0 else "#b58863"

                self.buttons[r][c].config(
                    text=text,
                    font=("Arial", 24),
                    bg=bg
                )

        # highlight selected
        if self.selected:
            r, c = self.selected
            self.buttons[r][c].config(bg="yellow")

        # highlight legal moves
        for (r, c) in self.legal_moves:
            self.buttons[r][c].config(bg="light blue")

        # highlight check
        ck = self.game_state.king_in_check_rc()
        if ck:
            r, c = ck
            self.buttons[r][c].config(bg="red")

    # handle input
    def on_click(self, r, c):
        if self.game_state.is_game_over():
            self.show_game_over()
            return
        turn = self.game_state.board.turn == CHESS_TURN

        # block player input in AI vs AI
        if self.mode in ("mvm", "rvm", "rml", "mml"):
            return
        # block player when it's AI turn
        if self.mode == "pvr" and not turn:
            return
        if self.mode == "pvm" and not turn:
            return
        if self.mode == "pml" and not turn:
            return

        if self.selected is None:
            self.selected = (r, c)
            self.legal_moves = self.game_state.legal_moves_from(r, c)
        else:
            moved = self.game_state.make_move(self.selected, (r, c))
            if moved:
                print("Player move:", self.game_state.moveLog[-1])
            self.selected = None
            self.legal_moves = []

        self.update_board()

     # game over
    def show_game_over(self):
        result = self.game_state.game_result()
        messagebox.showinfo("Game Over", result)

    # AI turn
    def ai_loop(self):
        if self.game_state.is_game_over():
            self.show_game_over()
            return

        turn = self.game_state.board.turn == CHESS_TURN
        ai_type = None

        if self.mode == "pvp":
            ai_type = None
        elif self.mode == "pvr":
            ai_type = "random" if not turn else None
        elif self.mode == "pvm":
            ai_type = "minimax" if not turn else None
        elif self.mode == "rvm":
            ai_type = "random" if turn else "minimax"
        elif self.mode == "mvm":
            ai_type = "minimax"
        elif self.mode == "pml":
            ai_type = "ml" if not turn else None
        elif self.mode == "rml":
            ai_type = "random" if turn else "ml"
        elif self.mode == "mml":
            ai_type = "minimax" if turn else "ml"

        if ai_type:
            mv = None
            if ai_type == "random":
                mv = solver.random_move(self.game_state.board)
            elif ai_type == "minimax":
                prev_use = getattr(solver, 'USE_ML', True)
                solver.USE_ML = False
                mv = solver.find_best_move(self.game_state.board, MINIMAX_DEPTH)
                solver.USE_ML = prev_use
            elif ai_type == "ml":
                prev_use = getattr(solver, 'USE_ML', True)
                solver.USE_ML = True
                mv = solver.find_best_move(self.game_state.board, MINIMAX_DEPTH)
                solver.USE_ML = prev_use

            if mv:
                self.game_state.apply_move_obj(mv)
                print(f"AI ({ai_type}):", mv.uci())

        self.update_board()
        self.root.after(AI_DELAY, self.ai_loop)


def main():
    root = tk.Tk()
    app = ChessUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()