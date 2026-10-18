"""Puzzle records use immutable tuples after loading and validation."""

import unittest
from pathlib import Path

from connections_rl import Group, Puzzle, load_puzzle


EXAMPLE = Path(__file__).resolve().parents[1] / "data/puzzles/example.json"


class PuzzleTests(unittest.TestCase):
    def test_json_loader_converts_lists_to_tuples(self) -> None:
        puzzle = load_puzzle(EXAMPLE)
        self.assertEqual(puzzle.puzzle_id, "example-001")
        self.assertIsInstance(puzzle.groups, tuple)
        self.assertTrue(all(isinstance(group.words, tuple) for group in puzzle.groups))

    def test_direct_construction_rejects_mutable_lists(self) -> None:
        with self.assertRaisesRegex(ValueError, "group words must be a tuple"):
            Group("Colors", "yellow", ["RED", "BLUE", "GREEN", "GOLD"])

        puzzle = load_puzzle(EXAMPLE)
        with self.assertRaisesRegex(ValueError, "puzzle groups must be a tuple"):
            Puzzle(puzzle.puzzle_id, list(puzzle.groups))


if __name__ == "__main__":
    unittest.main()
