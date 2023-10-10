import numpy as np
from tqdm import tqdm

import openai


# Generate all 3x3 boards containing 0 or 1
def generate_states():
    states = []
    for i in range(2**9):
        state = np.array([int(c) for c in f"{i:09b}"]).reshape((3, 3))
        states.append(state)
    return states


# For a given board, compute the alive state of the center cell according to Conway's game of life
def next_state(state):
    center = state[1, 1]
    neighbors = state.sum() - center
    if center == 1 and neighbors in [2, 3]:
        return 1
    elif center == 0 and neighbors == 3:
        return 1
    else:
        return 0


def predict_next_state(state, model="gpt-4"):
    state_str = "\n".join(
        "".join("1" if state[x, y] else "0" for y in range(3)) for x in range(3)
    )
    state_tokens = "\n".join([" " + " ".join(line) for line in state_str.splitlines()])

    messages = [
        {
            "role": "system",
            "content": f"You are simulating conway's game of life. The rules: A cell is updated to alive if it has exactly 2 or exactly 3 live neighbors, and becomes or remains dead otherwise."
            "You are given the state of a cell and its 8 neighbors. Live cells are represented by ' 1' and dead cells are represented by ' 0'. The center cell is the cell you are predicting the next state of."
            "Count the number of alive and dead neighbors. Note the current state of the cell. Then apply the rule to output the next state of the center cell."
            "IMPORTANT: Only output the updated state of the center cell, as a 0 or a 1, not any other cells. Do not output anything else.",
        },
        {"role": "user", "content": state_tokens},
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )

    predicted_str = response.choices[0].message.content
    predicted_str.strip()

    predicted = int(predicted_str)
    return predicted


def main():
    states = generate_states()

    states_file = open("states.txt", "w")
    simulated_next_states_file = open("simulated_next_states.txt", "w")
    predicted_next_states_file = open("predicted_next_states.txt", "w")

    n_correct = 0
    state_transitions = np.zeros((2, 2))
    predicted_transitions = np.zeros((2, 2))
    for state in tqdm(states):
        current_state = state[1, 1]

        simulated = next_state(state)
        state_transitions[current_state, simulated] += 1

        predicted = predict_next_state(state)
        predicted_transitions[current_state, predicted] += 1

        states_file.write(f"{state}\n")
        simulated_next_states_file.write(f"{simulated}\n")
        predicted_next_states_file.write(f"{predicted}\n")

        if simulated == predicted:
            n_correct += 1

    print(f"Accuracy: {n_correct / len(states)}")
    print(f"State transitions: \n{state_transitions/len(states)}")
    print(f"Predicted transitions: \n{predicted_transitions/len(states)}")


if __name__ == "__main__":
    main()
