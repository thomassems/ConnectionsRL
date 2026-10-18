"""Chronological splitting keeps the held-out test puzzles separate."""

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from scripts.make_splits import make_splits


class SplitTests(unittest.TestCase):
    def test_chronological_splits_are_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            days = ("2024-01-01", "2024-10-01", "2025-03-01")
            for day in days:
                (root / f"nyt-{day}.json").write_text(json.dumps({
                    "puzzle_id": f"nyt-{day}",
                    "groups": [
                        {"title": str(index), "color": color,
                         "words": [f"{index}-{word}" for word in "abcd"]}
                        for index, color in enumerate(("yellow", "green", "blue", "purple"))
                    ],
                }))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "count": 3, "puzzles": [{"date": day} for day in days],
            }))
            result = make_splits(manifest, date(2024, 9, 30), date(2025, 2, 19))
            self.assertEqual(result["splits"], {
                "train": ["nyt-2024-01-01"],
                "validation": ["nyt-2024-10-01"],
                "test": ["nyt-2025-03-01"],
            })


if __name__ == "__main__":
    unittest.main()
