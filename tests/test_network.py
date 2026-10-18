"""Network features must not reveal the answer grouping."""

import importlib.util
import unittest

from connections_rl.agent_input import ClueBatch, WordKnowledge
from connections_rl.puzzle import Group, Puzzle


@unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("torch"),
                     "training dependencies are optional")
class NetworkTests(unittest.TestCase):
    def test_feedback_diagnostics_track_rank_and_repeated_overlap(self):
        from connections_rl.train import summarize_diagnostics

        before = {
            "feedback": "incorrect", "guess": ["A", "B", "C", "D"],
            "true_group_ranks": {
                "yellow": {"rank": 2}, "green": {"rank": 20},
                "blue": {"rank": 200}, "purple": {"rank": 1000},
            },
        }
        after = {
            "feedback": "correct", "guess": ["A", "B", "C", "E"],
            "true_group_ranks": {
                "yellow": {"rank": 1}, "green": {"rank": 15},
                "blue": {"rank": 180}, "purple": {"rank": 900},
            },
        }
        summary = summarize_diagnostics([{
            "groups_solved": 1, "diagnostic_turns": [before, after],
        }])
        self.assertEqual(summary["initial_true_groups_top_10"], 1)
        self.assertEqual(summary["first_guess_feedback"]["incorrect"], 1)
        followup = summary["wrong_guess_followups"]["incorrect"]
        self.assertEqual(followup["next_guess_overlap_at_least_3"], 1)
        self.assertEqual(followup["mean_best_true_rank_improvement"], 1)

    def test_candidate_features_do_not_depend_on_hidden_groups(self):
        import numpy as np

        from connections_rl.environment import ConnectionsEnv
        from connections_rl.network import build_features

        class Knowledge:
            def lookup(self, word):
                return WordKnowledge(word, (float(ord(word) - 65),) * 300)

        class Provider:
            def __init__(self, channel):
                self.channel = channel

            def find(self, words):
                return ClueBatch(self.channel, (True,) * 4, ())

        words = tuple(chr(65 + index) for index in range(16))
        colors = ("yellow", "green", "blue", "purple")
        first = Puzzle("a", tuple(Group(str(i), color, words[i * 4:(i + 1) * 4])
                                  for i, color in enumerate(colors)))
        second = Puzzle("b", tuple(Group(str(i), color, words[i::4])
                                   for i, color in enumerate(colors)))
        args = (Knowledge(), Provider("meaning"), Provider("word_form"))
        a = build_features(first, *args)
        b = build_features(second, *args)
        self.assertEqual(a.words, b.words)
        np.testing.assert_array_equal(a.candidates, b.candidates)
        np.testing.assert_array_equal(a.clues, b.clues)
        np.testing.assert_array_equal(a.word_vectors, b.word_vectors)
        env = ConnectionsEnv(first)
        initial = env.reset(seed=7)
        self.assertEqual(len(a.legal_indices(initial)), 1820)
        for group in first.groups:
            last = env.step(group.words).observation
        self.assertEqual(last.status, "won")
        self.assertEqual(len(a.legal_indices(last)), 0)


if __name__ == "__main__":
    unittest.main()
