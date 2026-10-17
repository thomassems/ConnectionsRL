# Connections RL

A reinforcement learning project for training an agent to solve NYT Connections puzzles.

The goal is to explore how an RL agent can learn to identify semantic relationships, word patterns, and other connections between groups of words.

## Project structure

- `connections_rl/puzzle.py` defines and validates the puzzle data shape.
- `connections_rl/environment.py` runs one puzzle episode and applies feedback and rewards.
- `connections_rl/rewards.py` defines the configurable reward values.
- `data/puzzles/` stores JSON puzzle records. `example.json` is a synthetic example, not a historical NYT puzzle.
- `tests/` checks the episode rules using the synthetic puzzle.

The neural network and historical puzzle dataset have not been added yet.

## Puzzle data

Each JSON record has a unique `puzzle_id` and exactly four groups. Each group has a category `title`, one of the colors `yellow`, `green`, `blue`, or `purple`, and exactly four `words`. All 16 words must be distinct. The environment will hold the groups as a private answer key and will shuffle the words before showing them to the agent. It will reveal a group's words, title, and color only after a correct guess. The answers will not be revealed after a failed game.

```json
{
  "puzzle_id": "example-001",
  "groups": [
    {"title": "Colors", "color": "yellow", "words": ["RED", "BLUE", "GREEN", "GOLD"]},
    {"title": "Animals", "color": "green", "words": ["DOG", "CAT", "HORSE", "SHEEP"]},
    {"title": "Fruits", "color": "blue", "words": ["APPLE", "PEAR", "BANANA", "PEACH"]},
    {"title": "Shapes", "color": "purple", "words": ["CIRCLE", "SQUARE", "TRIANGLE", "OVAL"]}
  ]
}
```

Python loads this format into the `Puzzle` and `Group` types with `connections_rl.load_puzzle`. Definitions, embeddings, and phrase data will live in a separate word-level cache so they can be reused across puzzles. Puzzle dates and numbers can be added when historical puzzles are collected.

## Training environment

`ConnectionsEnv` holds the groups as a private answer key. `reset()` shuffles the 16 words and returns an observation with remaining words, solved groups, mistakes left, complete guess history, and game status. It does not expose unsolved categories. A correct guess reveals that group's words, title, and color. A wrong guess reports whether it was one away, without revealing which group was nearly matched. Failure does not reveal the remaining answers.

The agent selects four remaining words per step. `available_guesses()` lists the legal combinations and excludes previously guessed combinations. A repeated guess, a removed word, or any other invalid guess raises `ValueError` without consuming a turn. The agent must submit the last four words after solving the first three groups to finish the puzzle.

| Result | Reward |
| --- | ---: |
| Correct group | +1 |
| One away | -0.2 |
| Other wrong guess | -1 |
| Fourth wrong guess (failure) | -5 only |
| Fourth correct group (completion) | +1 group reward and +5 bonus |

These are starting values and can be changed through `RewardConfig`.

```python
from connections_rl import ConnectionsEnv, load_puzzle

puzzle = load_puzzle("data/puzzles/example.json")
env = ConnectionsEnv(puzzle)
observation = env.reset(seed=7)
guess = env.available_guesses()[0]
result = env.step(guess)
print(result.reward, result.observation.guess_history[-1].feedback)
```

Run the checks with `python3 -m unittest discover -s tests -v`.
