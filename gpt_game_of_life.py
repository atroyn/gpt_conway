import numpy as np
import openai
from tqdm import tqdm

import os
import time


def update(board):
    """Update the board based on Conway's Game of Life rules."""
    new_board = board.copy()
    rows, cols = board.shape

    for x in range(rows):
        for y in range(cols):
            # Count the 8 neighbors
            neighbors = (
                np.sum(
                    board[
                        max(x - 1, 0) : min(x + 2, rows),
                        max(y - 1, 0) : min(y + 2, cols),
                    ]
                )
                - board[x, y]
            )

            # Apply the rules of Conway's Game of Life
            if board[x, y] and not 2 <= neighbors <= 3:
                new_board[x, y] = 0
            elif not board[x, y] and neighbors == 3:
                new_board[x, y] = 1

    return new_board


def render_board(board, display=False):
    """Turn the board into a string."""
    rows, cols = board.shape
    board_str = "\n".join(
        "".join("X" if board[x, y] else "." for y in range(cols)) for x in range(rows)
    )

    if display:
        os.system("clear")
        print(board_str)
        time.sleep(0.1)

    return board_str


def predict_board(board, height, width, model="gpt-4"):
    board_str = render_board(board)

    board_tokens = "\n".join([" " + " ".join(line) for line in board_str.splitlines()])

    messages = [
        {
            "role": "system",
            "content": f"You are simulating conway's game of life. You are given the board state. Live cells are represented by ' X' and dead cells are represented by ' .'. You will be shown a board state and you must predict the next board state. Assume all cells outside the board are dead."
            f"IMPORTANT: Only output the next board state, in the given representation. Only output the state of the cells in the initial board with width {width} and height {height}, not any other cells. Do not output anything else.",
        },
        {"role": "user", "content": board_tokens},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )

    # Transform the response into a board
    predicted_str = response.choices[0].message.content

    # Strip the space characters
    predicted_str = predicted_str.replace(" ", "")

    # Throws ValueError if the response is not a valid board
    predicted_board = np.array(
        [[1 if c == "X" else 0 for c in line] for line in predicted_str.split("\n")]
    )

    if predicted_board.shape != (height, width):
        raise ValueError("Wrong shape")

    return predicted_board


def main():
    n_trials = 1000

    height, width = 10, 10
    n_different = 0
    n_total = 0

    n_invalid = 0

    # Open and append to the files
    generated_file = open("generated.txt", "a")
    simulated_file = open("simulated.txt", "a")
    predicted_file = open("predicted.txt", "a")

    for trial in tqdm(range(n_trials)):
        # Generate a random board
        board = np.random.choice([0, 1], size=(height, width))

        # Write the board to the file
        generated_file.write(f"{render_board(board)}\n\n")

        # Run the game of life for one step
        simulated_board = update(board)
        simulated_file.write(f"{render_board(board)}\n\n")

        # Use the model to predict the board for one step
        try:
            predicted_board = predict_board(board=board, width=width, height=height)
        except ValueError:
            # Skip invalid boards
            n_invalid += 1
            predicted_file.write(f"Invalid board\n\n")
            continue

        predicted_file.write(f"{render_board(predicted_board)}\n\n")

        # Compute the number of cells that are different
        n_different += np.sum(np.abs(simulated_board - predicted_board))
        n_total += height * width

    print(f"Accuracy: {1 - n_different / n_total}")
    print(f"Invalid boards: {n_invalid / n_trials}")


if __name__ == "__main__":
    main()
