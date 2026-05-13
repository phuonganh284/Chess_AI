import pandas as pd
import chess


piece_to_index = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}


def get_column_names():
    columns = []
    piece_symbols = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
    
    # 768 cột cho 64 ô cờ (64 * 12)
    for i in range(64):
        for p in piece_symbols:
            columns.append(f"sq{i}_{p}")
            
    # 5 cột Meta-data
    columns.extend([
        "turn_white",
        "white_kingside_castle",
        "white_queenside_castle",
        "black_kingside_castle",
        "black_queenside_castle"
    ])
    return columns

def board_to_vector(board):
    vector = []
    
    for square in chess.SQUARES:
        square_vector = [0] * 12 
        piece = board.piece_at(square)
        if piece is not None:
            idx = piece_to_index[piece.symbol()]
            square_vector[idx] = 1 
        vector.extend(square_vector)
        
    vector.append(1 if board.turn == chess.WHITE else 0)
    
    vector.append(1 if board.has_kingside_castling_rights(chess.WHITE) else 0)
    vector.append(1 if board.has_queenside_castling_rights(chess.WHITE) else 0)
    vector.append(1 if board.has_kingside_castling_rights(chess.BLACK) else 0)
    vector.append(1 if board.has_queenside_castling_rights(chess.BLACK) else 0)
    
    return vector

def clean_dataset(input_csv, output_csv):
    print("Reading original dataset...")
    df = pd.read_csv(input_csv)
    
    X = []
    y = []
    total = len(df)
    
    print("Processing FENs to Vectors...")
    for index, row in df.iterrows():
        try:
            fen = row["FEN"]
            score = row["Analysis"]
            
            if isinstance(score, str) and not score.replace('.', '', 1).lstrip('-').isdigit():
                continue
                
            score = float(score) / 100.0
            
            board = chess.Board(fen)
            
            vector = board_to_vector(board)
            
            X.append(vector)
            y.append(score)
            
            if (index + 1) % 5000 == 0:
                print(f"Processed {index + 1}/{total}")
                
        except Exception as e:
            print(f"Skipped row {index}: {e}")

    print("Building Dataframe...")
    columns = get_column_names()
    cleaned_df = pd.DataFrame(X, columns=columns)
    cleaned_df["score"] = y

    print("Saving to CSV...")
    cleaned_df.to_csv(output_csv, index=False)
    print(f"Saved cleaned dataset to: {output_csv}")
    print(f"Dataset shape: {cleaned_df.shape} (Rows, Columns)")

if __name__ == "__main__":
    clean_dataset(
        "fen_analysis.csv",
        "chess_dataset_v2.csv"
    )