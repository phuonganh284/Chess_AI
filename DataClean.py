import pandas as pd
import chess

piece_map = {
    'P': 1,
    'N': 2,
    'B': 3,
    'R': 4,
    'Q': 5,
    'K': 6,
    'p': -1,
    'n': -2,
    'b': -3,
    'r': -4,
    'q': -5,
    'k': -6
}


def board_to_vector(board):
    vector = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            vector.append(0)
        else:
            vector.append(piece_map[piece.symbol()])
    return vector


# clean dataset
def clean_dataset(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    X = []
    y = []
    total = len(df)
    for index, row in df.iterrows():
        try:
            fen = row["FEN"]
            score = row["Analysis"]
            if isinstance(score, str):
                continue
            # convert centipawn -> pawn
            score = float(score) / 100.0
            # create board from FEN
            board = chess.Board(fen)
            # convert board -> vector
            vector = board_to_vector(board)
            X.append(vector)
            y.append(score)
            if (index + 1) % 1000 == 0:
                print(f"Processed {index + 1}/{total}")
        except Exception as e:
            print("Skipped row:", e)

    columns = []

    for i in range(64):
        columns.append(f"square_{i}")

    cleaned_df = pd.DataFrame(X, columns=columns)
    cleaned_df["score"] = y

    cleaned_df.to_csv(output_csv, index=False)
    print("Saved cleaned dataset to:", output_csv)


if __name__ == "__main__":

    clean_dataset(
        "fen_analysis.csv",
        "chess_dataset.csv"
    )