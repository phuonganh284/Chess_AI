import random
import chess
import math
import pickle
import pandas as pd
import DataClean


with open("chess_model.pkl", "rb") as f:
    ml_model = pickle.load(f)

FEATURE_COLUMNS = DataClean.get_column_names()

# Track weak opening patterns to avoid repeating them
weak_opening_moves = {}  # {fen_after_n_moves: (move_uci, loss_count)}


# Random move agent ----------------------------------------------------
def random_move(board):
    moves = list(board.legal_moves)
    if len(moves) == 0:
        return None
    return random.choice(moves)


# Simple evaluation -------------------------------------
pieceScore = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

REPETITION_SOFT_PENALTY = 0.25
REPETITION_DRAW_PENALTY = 0.60
WEAK_OPENING_PENALTY = 1.5  # Penalty for moves that led to losses in opening


def repetition_penalty(board):
    penalty = 0.0
    if board.is_repetition(2):
        penalty += REPETITION_SOFT_PENALTY
    if board.can_claim_threefold_repetition():
        penalty += REPETITION_DRAW_PENALTY

    if penalty == 0.0:
        return 0.0

    # Positive score favors White; penalize the side to move when repeating.
    return -penalty if board.turn == chess.WHITE else penalty


# white -> maximize
# black -> minimize
def evaluate(board):
    if board.is_checkmate():
        if board.turn:   # white turn -> white gets checked -> white lost
            return -9999
        else:            # black turn -> black gets checked -> black lost
            return 9999
    if board.is_stalemate():
        return 0
    whiteScore = 0
    blackScore = 0
    for piece in pieceScore:
        value = pieceScore[piece]
        whiteScore += len(board.pieces(piece, chess.WHITE))*value
        blackScore += len(board.pieces(piece, chess.BLACK))*value

    # Piece value calculation
    score = whiteScore - blackScore
    # Mobility calculation
    temp = board.copy()
    temp.turn = chess.WHITE
    whiteMobility = len(list(temp.legal_moves))
    temp.turn = chess.BLACK
    blackMobility = len(list(temp.legal_moves))

    score += 0.01 * (whiteMobility - blackMobility)
    score += repetition_penalty(board)
    return score

# Tactical threat detection for ML ---------
def tactical_threat_penalty(board):
    """
    Detect immediate tactical threats that ML model might miss:
    - Check: penalize hard
    - Hanging pieces: penalize
    - Mate threat next move: extreme penalty
    """
    threat_score = 0.0
    
    # Current player in check: very bad (-8)
    if board.is_check():
        threat_score = -8.0 if board.turn == chess.WHITE else 8.0
        
    # Detect hanging pieces (piece can be captured and not defended)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
            
        # Only check opponent pieces (pieces that just moved into position)
        attacker_color = piece.color
        piece_value = pieceScore.get(piece.piece_type, 0)
        
        if piece_value > 0:  # Has material value
            attackers = board.attackers(not attacker_color, square)
            defenders = board.attackers(attacker_color, square)
            
            if len(attackers) > len(defenders) and piece.piece_type != chess.KING:
                # Hanging piece: penalize side that owns it
                hanging_penalty = piece_value * 0.5
                threat_score += (-hanging_penalty if attacker_color == chess.WHITE else hanging_penalty)
    
    # Detect mate threat (opponent can mate in 1)
    opponent_board = board.copy()
    opponent_board.turn = not board.turn
    for move in opponent_board.legal_moves:
        opponent_board.push(move)
        if opponent_board.is_checkmate():
            # Mate threat detected: extreme penalty
            threat_score += (-15.0 if board.turn == chess.WHITE else 15.0)
            opponent_board.pop()
            break
        opponent_board.pop()
    
    return threat_score

# ML evaluation with threat detection -------
eval_cache = {}

def ml_evaluate(board):
    if board.is_checkmate():
        return -9999 if board.turn else 9999
    if board.is_stalemate():
        return 0

    fen = board.fen()
    if fen in eval_cache:
        return float(eval_cache[fen] + repetition_penalty(board) + tactical_threat_penalty(board))

    vec = DataClean.board_to_vector(board)
    vec_df = pd.DataFrame([vec], columns=FEATURE_COLUMNS)
    
    pred = ml_model.predict(vec_df)[0]

    eval_cache[fen] = pred

    # Combine ML score with tactical threat detection
    threat_adj = tactical_threat_penalty(board)
    final_score = float(pred + repetition_penalty(board) + threat_adj)
    
    return final_score

""" 
def hybrid_evaluate(board):
    # Hybrid: Combine heuristic (reliable) + ML (learning pattern)
    # Heuristic: 55% weight (tactical, stable), ML: 45% weight (strategic learning)
    # ML now includes tactical threat detection
    heuristic_score = evaluate(board)
    ml_score = ml_evaluate(board)
    
    # Normalize but allow tactical threats to show
    # Clamp heuristic to [-50, 50], ML already includes threats up to ±15
    h_clamped = max(-50, min(50, heuristic_score))
    m_clamped = max(-50, min(50, ml_score))  # Wider range to let threats shine
    
    hybrid_score = 0.55 * h_clamped + 0.45 * m_clamped
    return hybrid_score
"""
# Minimax  -------------------------------------
def minimax(board, depth, alpha, beta, maximizingPlayer, use_ml):
    # base case
    if board.is_game_over() or depth == 0:
        if use_ml:
            return ml_evaluate(board)  # Use hybrid instead of pure ML
        else:
            return evaluate(board)
            
    moves = list(board.legal_moves)
    # white turn (maximize)
    if maximizingPlayer:
        maxEval = -math.inf
        moves.sort(key=lambda move: board.is_capture(move), reverse=True)
        for move in moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False, use_ml)
            board.pop()
            maxEval = max(eval, maxEval)
            alpha = max(alpha, maxEval)
            if beta <= alpha:
                break
            if maximizingPlayer and maxEval == 9999:
                break
        return maxEval
    # black turn (minimize)
    else:
        minEval = math.inf
        moves.sort(key=lambda move: board.is_capture(move), reverse=True)
        for move in moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True, use_ml)
            board.pop()
            minEval = min(eval, minEval)
            beta = min(beta, minEval)
            if beta <= alpha:
                break
            if not maximizingPlayer and minEval == -9999:
                break
        return minEval
        

def find_best_move(board, depth, use_ml_eval=False):
    best_moves = []
    maximizingPlayer = (board.turn == chess.WHITE) # if agent is white
    eps = 1e-9
    
    # use_ml_eval: True = use ML evaluation, False = use heuristic evaluation
    use_ml = use_ml_eval
    
    if maximizingPlayer:
        best_value = -math.inf
    else:
        best_value = math.inf

    for move in board.legal_moves:
        board.push(move)
        value = minimax(board, depth - 1, -100000, 100000, not maximizingPlayer, use_ml)
        
        # Apply weak opening pattern penalty in early game (< 10 moves)
        if use_ml and len(board.move_stack) <= 10:
            fen_key = board.fen()
            if fen_key in weak_opening_moves:
                loss_count = weak_opening_moves[fen_key][1]
                value -= WEAK_OPENING_PENALTY * min(loss_count, 2)  # Cap penalty at 2 losses
        
        board.pop()

        if maximizingPlayer:
            if value > best_value + eps:
                best_value = value
                best_moves = [move]
            elif abs(value - best_value) <= eps:
                best_moves.append(move)
        else:
            if value < best_value - eps:
                best_value = value
                best_moves = [move]
            elif abs(value - best_value) <= eps:
                best_moves.append(move)

    if not best_moves:
        return None
    
    # Early-game randomization for ML: if many moves tied, pick random from top candidates
    if use_ml and len(board.move_stack) <= 8 and len(best_moves) > 1:
        # Increase diversity in opening by sometimes picking non-first best move
        return random.choice(best_moves[:min(3, len(best_moves))])
    
    return random.choice(best_moves)