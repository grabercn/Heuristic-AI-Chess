# Heuristic-AI-Chess

Heuristic-AI-Chess is a Python-based chess game where players can compete against an AI. The AI uses basic heuristic evaluations and training data to make moves. You can play against the AI or run a parallelized training simulation to improve its decision-making capabilities.

## Features

- Play as white against the AI (which plays as black).
- The AI uses a heuristic evaluation function for board scoring.
- A progress bar with ETA for AI training simulations.
- Multi-threaded parallel training for faster AI training.
- Display of taken pieces and current AI heuristic score during gameplay.
- Precompiled `.exe` available for easy execution without needing Python installation.

## How to Play

1. **Launch the game** by running the precompiled `.exe` file.
2. **Main Menu Options**:
   - Press `1` to play against the AI.
   - Press `2` to train the AI using simulated games.

## Controls

- Click on a piece to select it (white pieces only).
- Click on a valid square to move the piece.
- The AI will automatically move after you make your move.
- The game will display the current AI heuristic score and track taken pieces for both sides.

## AI Training

- The AI training uses a random move strategy and tracks the number of wins, losses, and draws.
- Training can be parallelized across multiple CPU cores to speed up simulations.
- The training progress and ETA are displayed in a progress bar.
- You can adjust the number of games and workers by changing the `train_ai_parallel` function parameters in the code.

## Dependencies

This project requires the following Python libraries:

- `pygame`
- `chess`
- `concurrent.futures`
- `random`
- `pickle`

## Compiled Version

A precompiled .exe is available for download, allowing you to run the game without needing Python or any libraries installed.
Credits

Chess piece images and chessboard image sourced from the open resources provided by Wikipedia Chess Pieces and other free chess assets.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.


