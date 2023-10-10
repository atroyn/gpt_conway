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
