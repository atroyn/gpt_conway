import numpy as np
import matplotlib.pyplot as plt


def main():
    generated_file = open("generated.txt", "r")
    simulated_file = open("simulated.txt", "r")
    predicted_file = open("predicted.txt", "r")

    # Load the boards from the files. Each row is one line of the file. Boards are separated by newlines.
    generated_boards = [
        np.array([[int(c) for c in line] for line in board.split("\n")])
        for board in generated_file.read().split("\n\n")[:-1]
    ]
    simulated_boards = [
        np.array([[int(c) for c in line] for line in board.split("\n")])
        for board in simulated_file.read().split("\n\n")[:-1]
    ]
    predicted_boards = [
        np.array([[int(c) for c in line] for line in board.split("\n")])
        for board in predicted_file.read().split("\n\n")[:-1]
    ]

    # Compute the accuracy of the model against the simulation
    n_different = 0
    n_total = 0
    difference_board = np.zeros(generated_boards[0].shape)

    correct_count = np.zeros(2**9)
    predicted_alive = np.zeros(2**9)
    predicted_dead = np.zeros(2**9)
    totals = np.zeros(2**9)

    for generated_board, simulated_board, predicted_board in zip(
        generated_boards, simulated_boards, predicted_boards
    ):
        difference_board += np.abs(simulated_board - predicted_board)

        n_different += np.sum(np.abs(simulated_board - predicted_board))
        n_total += np.prod(simulated_board.shape)

        for i in range(1, generated_board.shape[0] - 1):
            for j in range(1, generated_board.shape[1] - 1):
                subgrid = generated_board[i - 1 : i + 2, j - 1 : j + 2]
                # Convert the subgrid to its integer index from its binary representation
                subgrid_index = int("".join(str(c) for c in subgrid.flatten()), 2)

                if predicted_board[i, j] == simulated_board[i, j]:
                    correct_count[subgrid_index] += 1

                if predicted_board[i, j] == 1:
                    predicted_alive[subgrid_index] += 1
                else:
                    predicted_dead[subgrid_index] += 1

                totals[subgrid_index] += 1

    print(f"Accuracy: {1 - n_different / n_total}")

    # Plot the difference board, with the legend indicating the number of boards that were different at each cell
    plt.imshow(difference_board, norm=plt.Normalize(0, len(generated_boards)))
    plt.colorbar()
    plt.show()

    for i in range(2**9):
        if totals[i] == 0:
            continue
        print(
            f"index: {i} Correct: {correct_count[i]/totals[i]}, Dead: {predicted_dead[i]/totals[i]}, Alive: {predicted_alive[i]/totals[i]}, Total: {totals[i]}"
        )

    # Plot a bar chart of the percentage of correct predictions per subgrid
    plt.bar(
        range(2**9),
        correct_count / totals,
        color="blue",
        label="Correct",
        alpha=0.5,
    )
    plt.show()

    # Compute the weighted entropy of the alive / dead predictions
    entropy = np.zeros(2**9)
    for i in range(2**9):
        if totals[i] == 0:
            continue
        p_alive = predicted_alive[i] / totals[i]
        p_dead = predicted_dead[i] / totals[i]

        # Compute the entropy, catching NaNs
        entropy[i] = (
            -p_alive * np.log(p_alive)
            if p_alive > 0
            else 0 - p_dead * np.log(p_dead)
            if p_dead > 0
            else 0
        )

    # Plot a bar chart of the entropy per subgrid
    plt.bar(range(2**9), entropy, color="red", label="Entropy", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()
