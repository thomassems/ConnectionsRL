"""Behavior checks for the puzzle episode and its public feedback."""

import unittest
from pathlib import Path

from connections_rl import ConnectionsEnv, load_puzzle


EXAMPLE = Path(__file__).resolve().parents[1] / "data/puzzles/example.json"


class ConnectionsEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = ConnectionsEnv(load_puzzle(EXAMPLE))
        self.initial = self.env.reset(seed=7)

    def test_reset_hides_answers_and_lists_valid_actions(self) -> None:
        self.assertEqual(len(self.initial.remaining_words), 16)
        self.assertEqual(self.initial.solved_groups, ())
        self.assertEqual(self.initial.guess_history, ())
        self.assertEqual(self.initial.mistakes_left, 4)
        self.assertEqual(self.initial.status, "in_progress")
        self.assertEqual(len(self.env.available_guesses()), 1820)
        self.assertEqual(self.env.reset(seed=7), self.initial)

    def test_wrong_feedback_and_repeated_guess(self) -> None:
        one_away = ("RED", "BLUE", "GREEN", "DOG")
        result = self.env.step(one_away)
        self.assertEqual(result.reward, -0.2)
        self.assertFalse(result.terminated)
        self.assertEqual(result.observation.mistakes_left, 3)
        self.assertEqual(result.observation.guess_history[-1].feedback, "one_away")
        self.assertIsNone(result.observation.guess_history[-1].revealed_group)
        self.assertNotIn(one_away, self.env.available_guesses())

        before = self.env.observation()
        with self.assertRaises(ValueError):
            self.env.step(tuple(reversed(one_away)))
        self.assertEqual(self.env.observation(), before)

        result = self.env.step(("RED", "BLUE", "CAT", "HORSE"))
        self.assertEqual(result.reward, -1.0)
        self.assertEqual(result.observation.guess_history[-1].feedback, "incorrect")

    def test_all_four_groups_must_be_submitted_to_win(self) -> None:
        groups = (
            ("RED", "BLUE", "GREEN", "GOLD"),
            ("DOG", "CAT", "HORSE", "SHEEP"),
            ("APPLE", "PEAR", "BANANA", "PEACH"),
            ("CIRCLE", "SQUARE", "TRIANGLE", "OVAL"),
        )
        titles = ("Colors", "Animals", "Fruits", "Shapes")
        colors = ("yellow", "green", "blue", "purple")
        for index, guess in enumerate(groups[:3]):
            result = self.env.step(guess)
            self.assertEqual(result.reward, 1.0)
            self.assertFalse(result.terminated)
            self.assertEqual(result.observation.guess_history[-1].feedback, "correct")
            self.assertEqual(result.observation.solved_groups[-1].words, guess)
            self.assertEqual(result.observation.solved_groups[-1].title, titles[index])
            self.assertEqual(result.observation.solved_groups[-1].color, colors[index])
            self.assertEqual(len(result.observation.remaining_words), 12 - 4 * index)

        self.assertEqual(result.observation.status, "in_progress")
        self.assertEqual(len(self.env.available_guesses()), 1)
        self.assertEqual(frozenset(self.env.available_guesses()[0]), frozenset(groups[3]))

        final = self.env.step(groups[3])
        self.assertEqual(final.reward, 6.0)
        self.assertTrue(final.terminated)
        self.assertEqual(final.observation.status, "won")
        self.assertEqual(final.observation.remaining_words, ())
        self.assertEqual(self.env.available_guesses(), ())
        with self.assertRaises(RuntimeError):
            self.env.step(groups[3])

    def test_fourth_wrong_guess_gets_only_failure_penalty(self) -> None:
        guesses = (
            ("RED", "BLUE", "DOG", "CAT"),
            ("RED", "BLUE", "HORSE", "SHEEP"),
            ("GREEN", "GOLD", "DOG", "CAT"),
            ("RED", "BLUE", "GREEN", "DOG"),
        )
        for guess in guesses[:3]:
            result = self.env.step(guess)
            self.assertEqual(result.reward, -1.0)
            self.assertFalse(result.terminated)

        final = self.env.step(guesses[3])
        self.assertEqual(final.reward, -5.0)
        self.assertTrue(final.terminated)
        self.assertEqual(final.observation.status, "lost")
        self.assertEqual(final.observation.mistakes_left, 0)
        self.assertEqual(len(final.observation.remaining_words), 16)
        self.assertEqual(final.observation.solved_groups, ())
        self.assertEqual(final.observation.guess_history[-1].feedback, "one_away")
        self.assertTrue(
            all(record.revealed_group is None for record in final.observation.guess_history)
        )
        self.assertEqual(self.env.available_guesses(), ())

    def test_invalid_guesses_do_not_change_state(self) -> None:
        invalid_guesses = (
            ("RED", "RED", "DOG", "CAT"),
            ("RED", "DOG", "CAT"),
            ("RED", "DOG", "CAT", "UNKNOWN"),
            "RED",
        )
        for invalid in invalid_guesses:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.env.step(invalid)
            self.assertEqual(self.env.observation(), self.initial)

        self.env.step(("RED", "BLUE", "GREEN", "GOLD"))
        before = self.env.observation()
        with self.assertRaises(ValueError):
            self.env.step(("RED", "DOG", "CAT", "HORSE"))
        self.assertEqual(self.env.observation(), before)


if __name__ == "__main__":
    unittest.main()
