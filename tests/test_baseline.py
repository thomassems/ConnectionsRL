"""The runnable baseline uses game feedback without seeing the answer key."""

import unittest

from connections_rl import ClueBatch, GuessRecord, Observation
from connections_rl.baseline import ClueBaseline


class EmptyMeanings:
    def find(self, words):
        return ClueBatch("meaning", (True,) * 4, ())

    def find_after_edits(self, words, forms, relation):
        return ClueBatch("meaning", (False,) * 4, ())


class EmptyForms:
    def find(self, words):
        return ClueBatch("word_form", (True,) * 4, ())


class BaselineTests(unittest.TestCase):
    def test_incorrect_feedback_rules_out_three_word_overlap(self):
        baseline = ClueBaseline(EmptyMeanings(), EmptyForms())
        wrong = GuessRecord(("A", "B", "C", "D"), "incorrect")
        observation = Observation(
            ("A", "B", "C", "D", "E", "F", "G", "H"),
            (), 3, (wrong,), "in_progress",
        )
        impossible = ("A", "B", "C", "E")
        possible = ("E", "F", "G", "H")

        ranked = baseline.rank(observation, (impossible, possible))

        self.assertEqual(tuple(item.words for item in ranked), (possible,))

    def test_one_away_feedback_gives_a_small_hint(self):
        baseline = ClueBaseline(EmptyMeanings(), EmptyForms())
        candidate = ("A", "B", "C", "E")
        history = (GuessRecord(("A", "B", "C", "D"), "one_away"),)

        self.assertEqual(baseline.score(candidate, history).value, 0.05)
        self.assertEqual(baseline.score(candidate).value, 0.0)


if __name__ == "__main__":
    unittest.main()
