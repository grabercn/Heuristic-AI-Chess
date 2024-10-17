import pygame
import chess
import chess.engine
import random
import pickle
import time
import concurrent.futures
from queue import Queue

# Pygame setup
pygame.init()
width, height = 640, 760  # Increase height for the stats area
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Chess Game with AI')

# Load chessboard image and resize it
board_image = pygame.image.load('resources/chessboard.png')
board_image = pygame.transform.scale(board_image, (640, 640))  # Board size

# Load and scale images of chess pieces
piece_images = {
    'P': pygame.transform.scale(pygame.image.load('resources/white-pawn.png'), (80, 80)),  # Updated path
    'N': pygame.transform.scale(pygame.image.load('resources/white-knight.png'), (80, 80)),  # Updated path
    'B': pygame.transform.scale(pygame.image.load('resources/white-bishop.png'), (80, 80)),  # Updated path
    'R': pygame.transform.scale(pygame.image.load('resources/white-rook.png'), (80, 80)),  # Updated path
    'Q': pygame.transform.scale(pygame.image.load('resources/white-queen.png'), (80, 80)),  # Updated path
    'K': pygame.transform.scale(pygame.image.load('resources/white-king.png'), (80, 80)),  # Updated path
    'p': pygame.transform.scale(pygame.image.load('resources/black-pawn.png'), (80, 80)),  # Updated path
    'n': pygame.transform.scale(pygame.image.load('resources/black-knight.png'), (80, 80)),  # Updated path
    'b': pygame.transform.scale(pygame.image.load('resources/black-bishop.png'), (80, 80)),  # Updated path
    'r': pygame.transform.scale(pygame.image.load('resources/black-rook.png'), (80, 80)),  # Updated path
    'q': pygame.transform.scale(pygame.image.load('resources/black-queen.png'), (80, 80)),  # Updated path
    'k': pygame.transform.scale(pygame.image.load('resources/black-king.png'), (80, 80))  # Updated path
}

def evaluate_board(board):
    # Material values for pieces
    piece_values = {
        'P': 1, 'N': 3.2, 'B': 3.33, 'R': 5, 'Q': 9, 'K': 0,
        'p': -1, 'n': -3.2, 'b': -3.33, 'r': -5, 'q': -9, 'k': 0
    }
    
    # Piece-square tables for positional evaluation
    pawn_table = [
        0, 0, 0, 0, 0, 0, 0, 0,
        0.5, 1, 1, -2, -2, 1, 1, 0.5,
        0.5, 0.5, 1, 1.5, 1.5, 1, 0.5, 0.5,
        0, 0, 0, 2, 2, 0, 0, 0,
        0.5, 0, 0, 1, 1, 0, 0, 0.5,
        0.5, 1, 1, -1, -1, 1, 1, 0.5,
        0, 0, 0, 0, 0, 0, 0, 0,
    ]
    
    knight_table = [
        -5, -4, -3, -3, -3, -3, -4, -5,
        -4, -2, 0, 0, 0, 0, -2, -4,
        -3, 0, 1, 1.5, 1.5, 1, 0, -3,
        -3, 0.5, 1.5, 2, 2, 1.5, 0.5, -3,
        -3, 0, 1.5, 2, 2, 1.5, 0, -3,
        -3, 0.5, 1, 1.5, 1.5, 1, 0.5, -3,
        -4, -2, 0, 0.5, 0.5, 0, -2, -4,
        -5, -4, -3, -3, -3, -3, -4, -5,
    ]
    
    bishop_table = [
        -2, -1, -1, -1, -1, -1, -1, -2,
        -1, 0, 0, 0, 0, 0, 0, -1,
        -1, 0, 0.5, 1, 1, 0.5, 0, -1,
        -1, 0.5, 0.5, 1, 1, 0.5, 0.5, -1,
        -1, 0, 1, 1, 1, 1, 0, -1,
        -1, 1, 1, 1, 1, 1, 1, -1,
        -1, 0.5, 0, 0, 0, 0, 0.5, -1,
        -2, -1, -1, -1, -1, -1, -1, -2,
    ]
    
    rook_table = [
        0, 0, 0, 0, 0, 0, 0, 0,
        0.5, 1, 1, 1, 1, 1, 1, 0.5,
        -0.5, 0, 0, 0, 0, 0, 0, -0.5,
        -0.5, 0, 0, 0, 0, 0, 0, -0.5,
        -0.5, 0, 0, 0, 0, 0, 0, -0.5,
        -0.5, 0, 0, 0, 0, 0, 0, -0.5,
        -0.5, 0, 0, 0, 0, 0, 0, -0.5,
        0, 0, 0, 0.5, 0.5, 0, 0, 0,
    ]
    
    queen_table = [
        -2, -1, -1, -0.5, -0.5, -1, -1, -2,
        -1, 0, 0, 0, 0, 0, 0, -1,
        -1, 0, 0.5, 0.5, 0.5, 0.5, 0, -1,
        -0.5, 0, 0.5, 0.5, 0.5, 0.5, 0, -0.5,
        0, 0, 0.5, 0.5, 0.5, 0.5, 0, -0.5,
        -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0, -1,
        -1, 0, 0.5, 0, 0, 0, 0, -1,
        -2, -1, -1, -0.5, -0.5, -1, -1, -2,
    ]
    
    king_middle_game_table = [
        -3, -4, -4, -5, -5, -4, -4, -3,
        -3, -4, -4, -5, -5, -4, -4, -3,
        -3, -4, -4, -5, -5, -4, -4, -3,
        -3, -4, -4, -5, -5, -4, -4, -3,
        -2, -3, -3, -4, -4, -3, -3, -2,
        -1, -2, -2, -2, -2, -2, -2, -1,
        2, 2, 0, 0, 0, 0, 2, 2,
        2, 3, 1, 0, 0, 1, 3, 2,
    ]
    
    king_end_game_table = [
        -5, -4, -3, -2, -2, -3, -4, -5,
        -3, -2, -1, 0, 0, -1, -2, -3,
        -3, -1, 2, 3, 3, 2, -1, -3,
        -3, -1, 3, 4, 4, 3, -1, -3,
        -3, -1, 3, 4, 4, 3, -1, -3,
        -3, -1, 2, 3, 3, 2, -1, -3,
        -3, -3, -1, 0, 0, -1, -3, -3,
        -5, -3, -3, -3, -3, -3, -3, -5,
    ]
    
    def get_piece_square_value(piece, square):
        piece_type = piece.symbol()
        if piece_type.lower() == 'p':
            table = pawn_table
        elif piece_type.lower() == 'n':
            table = knight_table
        elif piece_type.lower() == 'b':
            table = bishop_table
        elif piece_type.lower() == 'r':
            table = rook_table
        elif piece_type.lower() == 'q':
            table = queen_table
        elif piece_type.lower() == 'k':
            # Choose between middle-game and end-game evaluation
            table = king_middle_game_table if not is_endgame(board) else king_end_game_table
        else:
            return 0  # In case of unknown piece types

        if piece_type.isupper():
            return table[square]  # White pieces
        else:
            return -table[63 - square]  # Black pieces (board mirrored)

    def is_endgame(board):
        """Determine if the game has transitioned to the endgame."""
        queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
        minor_pieces = len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK)) + \
                       len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK))
        return queens == 0 or minor_pieces <= 3

    score = 0
    for square, piece in board.piece_map().items():
        score += piece_values[piece.symbol()]  # Add material value
        score += get_piece_square_value(piece, square)  # Add positional value

    # Add additional evaluations like pawn structure, mobility, and king safety here

    return score


def ai_move(board):
    # Load the AI training data
    try:
        with open("ai_training_data_parallel.pkl", "rb") as f:
            ai_training_data = pickle.load(f)
    except FileNotFoundError:
        print("Training data not found! AI will move randomly.")
        ai_training_data = {"wins": 0, "losses": 0, "draws": 0}

    legal_moves = list(board.legal_moves)
    
    # AI logic using training data
    move = None
    if ai_training_data["wins"] > ai_training_data["losses"]:
        # Aggressive strategy - favor captures
        capture_moves = [m for m in legal_moves if board.is_capture(m)]
        if capture_moves:
            move = random.choice(capture_moves)
        else:
            move = random.choice(legal_moves)
    else:
        # Defensive strategy - random legal move
        move = random.choice(legal_moves)

    return move

def draw_progress_bar(screen, progress, total, width=500, height=25, eta=0):
    """Draws a progress bar on the screen with ETA."""
    font = pygame.font.SysFont("Arial", 24)
    progress_label = font.render(f"Progress: {progress}/{total}", 1, (255, 255, 255))
    
    # Format ETA based on its value
    if eta < 60:
        eta_label = font.render(f"ETA: {eta:.2f} secs", 1, (255, 255, 255))
    elif eta < 3600:
        eta_minutes = eta / 60
        eta_label = font.render(f"ETA: {eta_minutes:.2f} mins", 1, (255, 255, 255))
    else:
        eta_hours = eta / 3600
        eta_label = font.render(f"ETA: {eta_hours:.2f} hours", 1, (255, 255, 255))

    # Draw the progress bar
    screen.blit(progress_label, (70, 30))
    screen.blit(eta_label, (70, 60))

    pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(70, 90, width, height), 2)  # Draw border
    inner_width = (progress / total) * width
    pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(70, 90, inner_width, height))   # Draw filled progress

def calculate_eta(start_time, games_completed, total_games):
    """Calculates the estimated time of completion (ETA) in seconds."""
    elapsed_time = time.time() - start_time
    avg_time_per_game = elapsed_time / games_completed if games_completed > 0 else 0
    games_left = total_games - games_completed
    eta = avg_time_per_game * games_left
    return eta

def simulate_game():
    """Simulate a game between the AI and an opponent with randomized heuristics."""
    # Define possible game outcomes
    outcomes = ["win", "loss", "draw"]
    
    # Add randomized heuristic logic
    # The heuristic could represent risk-taking behavior, defensive play, etc.
    ai_choice = random.choice(["offensive", "defensive", "random"])  # Random strategy selection
    
    # Based on the heuristic, change AI performance probabilities
    if ai_choice == "offensive":
        # AI plays aggressively, higher chance of winning but also losing
        return random.choices(outcomes, weights=[0.5, 0.4, 0.1])[0]
    elif ai_choice == "defensive":
        # AI plays defensively, higher chance of a draw or loss, but lower risk
        return random.choices(outcomes, weights=[0.3, 0.2, 0.5])[0]
    else:
        # Completely random moves with equal chances
        return random.choice(outcomes)

def train_ai_parallel(num_games, num_workers):
    """Train the AI using multiple cores, display progress, and avoid Pygame freezing."""
    ai_training_data = {"wins": 0, "losses": 0, "draws": 0}
    start_time = time.time()  # Track time to calculate ETA

    # Queue to track game results
    result_queue = Queue()

    def game_worker():
        """Worker function for parallel game simulations."""
        result = simulate_game()  # Simulate a game with randomized heuristics
        result_queue.put(result)

    # Start simulations in background threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(game_worker) for _ in range(num_games)]

        games_completed = 0
        while games_completed < num_games:
            try:
                result = result_queue.get(timeout=0.1)  # Get result with a timeout to allow Pygame to update
                games_completed += 1

                # Update training data
                if result == "win":
                    ai_training_data["wins"] += 1
                elif result == "loss":
                    ai_training_data["losses"] += 1
                else:
                    ai_training_data["draws"] += 1
                    
                print(result)

                # Calculate ETA and update progress bar
                eta = calculate_eta(start_time, games_completed, num_games)
                screen.fill((0, 0, 0))  # Clear the screen
                draw_progress_bar(screen, games_completed, num_games, eta=eta)
                pygame.display.flip()

                # Handle Pygame events to avoid freezing
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
            except Queue.Empty:
                pass

    # Save training data
    with open("ai_training_data_parallel.pkl", "wb") as f:
        pickle.dump(ai_training_data, f)

    # Final output
    font = pygame.font.SysFont("Arial", 24)
    screen.fill((0, 0, 0))
    label = font.render(f"Training Complete: Wins: {ai_training_data['wins']}, Losses: {ai_training_data['losses']}, Draws: {ai_training_data['draws']}", 1, (255, 255, 255))
    screen.blit(label, (50, 300))
    pygame.display.flip()
    time.sleep(3)

# Main Menu for selecting game mode
def main_menu():
    font = pygame.font.SysFont("Arial", 36)
    while True:
        screen.fill((0, 0, 0))
        label = font.render("Press 1 for Play vs AI, 2 for AI Training", 1, (255, 255, 255))
        screen.blit(label, (50, 300))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return "play"
                elif event.key == pygame.K_2:
                    return "train"

def play_vs_ai():
    board = chess.Board()
    running = True
    player_turn = True  # Assume player is always white
    selected_square = None  # To track selected piece
    valid_moves = []  # To track valid moves for the selected piece
    taken_pieces = []  # List to store taken pieces

    def draw_board():
        screen.blit(board_image, (0, 0))
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            row = square // 8
            col = square % 8
            # Draw pieces
            if piece:
                screen.blit(piece_images[piece.symbol()], (col * 80, row * 80))
            # Highlight valid moves
            if square in valid_moves:
                pygame.draw.rect(screen, (0, 255, 0, 50), (col * 80, row * 80, 80, 80))  # Green highlight

    def draw_score():
        font = pygame.font.SysFont("Arial", 24)
        score = evaluate_board(board)
        score_label = font.render(f"AI Heuristic Score: {str(round(score, 2))}", 1, (255, 255, 255))
        screen.blit(score_label, (10, 650))  # Position the score below the board

    def draw_taken_pieces():
        font = pygame.font.SysFont("Arial", 24)
        # Create separate lists for black and white taken pieces
        white_taken = [piece for piece in taken_pieces if piece.isupper()]  # Uppercase for white pieces
        black_taken = [piece for piece in taken_pieces if piece.islower()]  # Lowercase for black pieces

        # Position for white pieces
        white_label = font.render("White Taken Pieces:", 1, (255, 255, 255))
        screen.blit(white_label, (10, 680))  # Position the label for white pieces
        for index, piece in enumerate(white_taken):
            piece_label = font.render(piece, 1, (255, 255, 255))
            screen.blit(piece_label, (10 + index * 80, 710))  # Position each white piece horizontally

        # Position for black pieces, below white pieces
        black_label = font.render("Black Taken Pieces:", 1, (255, 255, 255))
        screen.blit(black_label, (10, 740))  # Position the label for black pieces
        for index, piece in enumerate(black_taken):
            piece_label = font.render(piece, 1, (255, 255, 255))
            screen.blit(piece_label, (10 + index * 80, 770))  # Position each black piece horizontally

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.MOUSEBUTTONDOWN and player_turn:
                # Player move
                x, y = pygame.mouse.get_pos()
                col = x // 80
                row = y // 80
                square = row * 8 + col

                # Check if a piece is clicked
                if selected_square is None:
                    piece = board.piece_at(square)
                    if piece and piece.color == chess.WHITE:  # Only allow white pieces to be selected
                        selected_square = square
                        # Calculate valid moves for the selected piece
                        valid_moves = [move.to_square for move in board.legal_moves if move.from_square == selected_square]
                else:
                    # If a square is clicked
                    if square in valid_moves:
                        # Move the piece
                        move = chess.Move(from_square=selected_square, to_square=square)
                        taken_piece = board.piece_at(square)  # Check if a piece was taken
                        if taken_piece:
                            taken_pieces.append(taken_piece.symbol())
                        board.push(move)
                        player_turn = False  # Switch turn to AI
                    # Reset selection
                    selected_square = None
                    valid_moves = []

        # AI Move
        if not player_turn and not board.is_game_over():
            ai_move_ = ai_move(board)
            if ai_move_:
                taken_piece = board.piece_at(ai_move_.to_square)  # Check if a piece was taken
                if taken_piece:
                    taken_pieces.append(taken_piece.symbol())
                board.push(ai_move_)
            player_turn = True

        # Redraw the board
        screen.fill((0, 0, 0))  # Clear the screen with black
        draw_board()
        draw_score()  # Draw the score
        draw_taken_pieces()  # Draw taken pieces
        pygame.display.flip()

        if board.is_game_over():
            running = False
            print("Game Over:")
            screen.fill((0, 0, 0))  # Clear the screen with black
            font = pygame.font.SysFont("Arial", 36)
            label = font.render("Game Over!", 1, (255, 255, 255))
            screen.blit(label, (50, 300))
            time.sleep(500)
            return

# Main loop
while True:
    mode = main_menu()
    if mode == "play":
        play_vs_ai()
    elif mode == "train":
        # select the training params here!
        train_ai_parallel(num_games=100000, num_workers=5)
        
        