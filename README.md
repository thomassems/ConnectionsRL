# Connections RL

A reinforcement learning project for training an agent to solve NYT Connections puzzles.

The goal is to explore how an RL agent can learn to identify semantic relationships, word patterns, and other connections between groups of words.

## Project structure

- `connections_rl/puzzle.py` defines and validates the puzzle data shape.
- `data/puzzles/` stores JSON puzzle records. `example.json` is a synthetic example, not a historical NYT puzzle.

The training environment and reward function are still to be designed.

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
