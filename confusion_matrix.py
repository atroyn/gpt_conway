import numpy as np


def main():
    simulated_file = open("simulated_next_states.txt", "r")
    predicted_file = open("predicted_next_states.txt", "r")

    # Compute the confusion matrix as percentages
    confusion_matrix = np.zeros((2, 2))
    for simulated_str, predicted_str in zip(simulated_file, predicted_file):
        simulated = int(simulated_str)
        predicted = int(predicted_str)

        confusion_matrix[simulated, predicted] += 1

    confusion_matrix /= np.sum(confusion_matrix, axis=1, keepdims=True)

    print(confusion_matrix)


if __name__ == "__main__":
    main()
