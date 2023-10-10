# GPT Game of life

This repo contains experiments intended to explore whether GPT has 'world models', using Conway's Game of Life as a testbed.

## World Models

A 'world model' is an explicit representation of a process which generates data. In contrast, 'surface statistics' represent a process as the empirical distribution of data it generates.
World models are desirable because they are more compact representations than surface statistics, and are a proxy for 'understanding' a process rather than just observing it.

A system with a world model should be able to apply that model consistently, across different scales and representations of the process. This is what we test here.

## Conway's Game of Life

Conway's Game of Life is a well-studied 2D grid cellular automaton, which evolves according to a simple set of rules:

- A cell is either 'alive' or 'dead'
- A cell's state in the next timestep is determined by the state of its 8 neighbours in the current timestep:
  - If a cell is alive, it remains alive if it has 2 or 3 live neighbours, otherwise it dies
  - If a cell is dead, it becomes alive if it has exactly 3 live neighbours, otherwise it remains dead

Game of life is a great testbed for our purposes because:

- The set of rules is simple and well-defined.
- The state of the board admits many representations.
- Even small boards (e.g. 10x10 boards as we use here) have combinatorially many (2^100) possible states, so a given board state is unlikely to be in the training set.
- The statistics of the state transitions are unbalanced.

To be very clear, I think it's possible to train a transformer model to play game of life, but that's not what we're doing here.

## Running the experiments

This code is pretty messy and ad-hoc, I might clean it up later.

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-YOUR_API_KEY

python gpt_game_of_life.py
python rules.py
```
