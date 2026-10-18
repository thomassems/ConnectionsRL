"""The first summary encoder keeps coverage and rare supporting evidence distinct."""

import unittest

from connections_rl import ClueBatch, ClueEvidence, ClueMatch, summarize_clues


class AgentInputTests(unittest.TestCase):
    def test_strong_rare_clue_can_be_the_best_match(self) -> None:
        rare = ClueEvidence(
            channel="meaning",
            relation="shared_meaning",
            raw_score=0.9,
            strength=0.9,
            matches=(ClueMatch("BANK", "river-bank", 0.05), ClueMatch("SHORE", "shore", 0.6)),
            explanation="River-edge meanings fit together",
        )
        common = ClueEvidence(
            channel="meaning",
            relation="shared_meaning",
            raw_score=0.4,
            strength=0.4,
            matches=(ClueMatch("BANK", "financial-bank", 0.95),),
            explanation="A common meaning with weaker group support",
        )
        batch = ClueBatch("meaning", (True, True, True, True), (common, rare))

        summary = summarize_clues(batch, ("shared_meaning", "shared_hypernym"))

        self.assertEqual(summary.coverage_count, 4)
        self.assertEqual(summary.evidence_count, 2)
        self.assertEqual(summary.best_strength, 0.9)
        self.assertEqual(summary.second_strength, 0.4)
        self.assertEqual(summary.best_support_count, 2)
        self.assertAlmostEqual(summary.best_priority_mean, 0.325)
        self.assertTrue(summary.best_priority_known)
        self.assertEqual(summary.specific, (("shared_meaning", 0.9), ("shared_hypernym", 0.0)))
        self.assertEqual(batch.evidence, (common, rare))

    def test_missing_source_is_distinct_from_no_match(self) -> None:
        missing = summarize_clues(ClueBatch("entity", (False,) * 4, ()))
        no_match = summarize_clues(ClueBatch("entity", (True,) * 4, ()))

        self.assertEqual(missing.evidence_count, no_match.evidence_count)
        self.assertEqual(missing.coverage_count, 0)
        self.assertEqual(no_match.coverage_count, 4)

    def test_wordnet_rank_and_tagged_count_stay_separate(self) -> None:
        evidence = ClueEvidence(
            channel="meaning",
            relation="shared_meaning",
            raw_score=0.8,
            strength=0.8,
            matches=(
                ClueMatch("BANK", "river-bank", sense_rank=2, tag_count=0),
                ClueMatch("SHORE", "shore", sense_rank=1, tag_count=7),
            ),
            explanation="River-edge meanings fit together",
        )

        summary = summarize_clues(ClueBatch("meaning", (True,) * 4, (evidence,)))

        self.assertEqual(summary.best_sense_rank_mean, 1.5)
        self.assertEqual(summary.best_tag_count_mean, 3.5)
        self.assertTrue(summary.best_sense_rank_known)
        self.assertTrue(summary.best_tag_count_known)
        self.assertFalse(summary.best_priority_known)


if __name__ == "__main__":
    unittest.main()
