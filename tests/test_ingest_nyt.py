"""Historical importer preserves answer groups and rejects bad color ordering."""

import unittest

from scripts.ingest_nyt import UnsupportedPuzzle, convert_official, validate_archive


class IngestTests(unittest.TestCase):
    def setUp(self):
        self.date = "2024-01-01"
        self.groups = [
            ("ONE", ["A", "B", "C", "D"]),
            ("TWO", ["E", "F", "G", "H"]),
            ("THREE", ["I", "J", "K", "L"]),
            ("FOUR", ["M", "N", "O", "P"]),
        ]
        self.payload = {
            "status": "OK", "id": 204, "print_date": self.date,
            "categories": [
                {"title": title, "cards": [{"content": word} for word in words]}
                for title, words in self.groups
            ],
        }

    def test_converts_official_response_and_checks_archive_colors(self):
        puzzle = convert_official(self.payload, self.date)
        archive = {
            self.date: {
                "date": self.date,
                "id": 203,  # Third-party numbering can disagree with NYT.
                "answers": [
                    {"level": index, "group": title, "members": words}
                    for index, (title, words) in enumerate(self.groups)
                ],
            }
        }

        validate_archive(puzzle, archive)

        self.assertEqual(puzzle.puzzle_id, "nyt-2024-01-01")
        self.assertEqual(tuple(group.color for group in puzzle.groups),
                         ("yellow", "green", "blue", "purple"))

    def test_rejects_archive_group_in_different_color(self):
        puzzle = convert_official(self.payload, self.date)
        archive = {
            self.date: {
                "answers": [
                    {"level": index, "members": words}
                    for index, (_, words) in enumerate(reversed(self.groups))
                ]
            }
        }

        with self.assertRaisesRegex(ValueError, "disagrees"):
            validate_archive(puzzle, archive)

    def test_excludes_image_cards_from_word_puzzles(self):
        self.payload["categories"][0]["cards"][0] = {"image_alt_text": "A"}
        with self.assertRaises(UnsupportedPuzzle):
            convert_official(self.payload, self.date)

    def test_excludes_symbol_only_puzzles(self):
        for category in self.payload["categories"]:
            for card in category["cards"]:
                card["content"] = "🍞"
        with self.assertRaises(UnsupportedPuzzle):
            convert_official(self.payload, self.date)


if __name__ == "__main__":
    unittest.main()
