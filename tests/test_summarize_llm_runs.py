"""The LLM run summarizer groups repeats and compares configurations."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_llm_runs import summarize


def _report(model, wins, purple_solved, *, provider="deepseek", effort=None,
            cost=0.1, ids=("a", "b", "c", "d")) -> dict:
    count = len(ids)
    return {
        "agent": "llm", "provider": provider, "model": model, "effort": effort,
        "count": count, "wins": wins, "win_rate": wins / count,
        "mean_groups_solved": 2.5, "mean_reward": 1.0,
        "manifest_sha256": "abc",
        "estimated_cost_usd": cost,
        "fallback_guesses": 0,
        "solved_rate_by_color": {
            color: {"solved": solved, "of": count, "rate": solved / count}
            for color, solved in (
                ("yellow", count), ("green", count), ("blue", count),
                ("purple", purple_solved),
            )
        },
        "results": [
            {"puzzle_id": pid, "status": "won" if index < wins else "lost"}
            for index, pid in enumerate(ids)
        ],
    }


class SummarizeTests(unittest.TestCase):
    def _write(self, directory, name, report):
        path = Path(directory) / name
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def test_repeats_of_one_config_are_averaged_with_spread(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = [
                self._write(tmp, "r1.json", _report("deepseek-chat", wins=2, purple_solved=1)),
                self._write(tmp, "r2.json", _report("deepseek-chat", wins=4, purple_solved=3)),
            ]
            result = summarize(paths)
            self.assertEqual(len(result["configurations"]), 1)
            row = result["configurations"][0]
            self.assertEqual(row["runs"], 2)
            self.assertEqual(row["wins"]["mean"], 3.0)
            self.assertEqual(row["wins"]["min"], 2)
            self.assertEqual(row["wins"]["max"], 4)
            self.assertGreater(row["wins"]["stdev"], 0.0)
            self.assertEqual(row["color_solve_rate"]["purple"]["mean"], 0.5)
            self.assertEqual(row["total_estimated_cost_usd"], 0.2)

    def test_win_consistency_counts_across_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Run 1 wins a,b; run 2 wins a,b,c -> a,b every run; c some; d none.
            r1 = _report("deepseek-chat", wins=2, purple_solved=0)
            r2 = _report("deepseek-chat", wins=3, purple_solved=0)
            paths = [self._write(tmp, "r1.json", r1), self._write(tmp, "r2.json", r2)]
            consistency = summarize(paths)["configurations"][0]["win_consistency"]
            self.assertEqual(consistency["won_by_every_run"], 2)
            self.assertEqual(consistency["won_by_some_run"], 3)
            self.assertEqual(consistency["won_by_no_run"], 1)

    def test_distinct_configs_form_a_leaderboard_sorted_by_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = [
                self._write(tmp, "chat.json", _report("deepseek-chat", wins=2, purple_solved=1)),
                self._write(tmp, "reason.json",
                            _report("deepseek-reasoner", wins=4, purple_solved=2)),
            ]
            rows = summarize(paths)["configurations"]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["model"], "deepseek-reasoner")  # most wins first

    def test_mismatched_puzzle_sets_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = [
                self._write(tmp, "a.json", _report("deepseek-chat", 2, 1, ids=("a", "b", "c", "d"))),
                self._write(tmp, "b.json", _report("deepseek-chat", 2, 1, ids=("a", "b", "c", "e"))),
            ]
            with self.assertRaises(ValueError):
                summarize(paths)


if __name__ == "__main__":
    unittest.main()
