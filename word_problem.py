from tqdm import tqdm

import openai


def next_state(neighbors_alive, neighbors_dead, self_alive):
    if self_alive and neighbors_alive in [2, 3]:
        return True
    elif not self_alive and neighbors_alive == 3:
        return True
    else:
        return False


# Generate all possible states in total form
def generate_states():
    states = []
    for i in range(8):
        for s in [True, False]:
            states.append(
                {
                    "alive": i,
                    "dead": 8 - i,
                    "self_alive": s,
                    "next": next_state(i, 8 - i, s),
                }
            )
    return states


def predict_word_problem(state, model="gpt-4"):
    messages = [
        {
            "role": "system",
            "content": f"You are simulating conway's game of life. A cell is updated to alive if it has exactly 2 or exactly 3 live neighbors, and becomes or remains dead otherwise."
            "Your task is to calculate the next state of a given cell, based on the number of live and dead neighbors it has, and its current state."
            "Output 1 if the cell is alive in the next time step, and 0 if it is dead. IMPORTANT: Only output the updated state of the center cell, as a 0 or a 1, not any other cells. Do not output anything else.",
        },
        {
            "role": "user",
            "content": f"Live neighbors: {state['alive']} , Dead neighbors: {state['dead']} , Current state: {1 if state['self_alive'] else 0}",
        },
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )

    # Transform the response into a board
    predicted_str = response.choices[0].message.content
    prediction = True if int(predicted_str) == 1 else False
    return prediction


def main():
    states = generate_states()

    word_problem_state_file = open("word_problem_states.txt", "w")
    word_problem_predictions_file = open("word_problem_predictions.txt", "w")

    n_correct = 0
    for state in tqdm(states):
        prediction = predict_word_problem(state)

        word_problem_state_file.write(f"{state}\n")
        word_problem_predictions_file.write(f"{prediction}\n")

        if prediction == state["next"]:
            n_correct += 1

    accuracy = n_correct / len(states)
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
