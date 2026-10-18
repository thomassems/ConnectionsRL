"""Meaning evidence keeps taxonomy, vectors, and spelling edits distinct."""

import unittest

from connections_rl import (
    ClueBatch,
    ClueEvidence,
    ClueMatch,
    Interpretation,
    MeaningClueProvider,
    SemanticAncestor,
    WordKnowledge,
)


class DictionaryCache:
    def __init__(self, records):
        self.records = records

    def lookup(self, word):
        return self.records.get(word)


def sense(key, ancestor, rank=1, count=0):
    return Interpretation(
        source="wordnet",
        source_id=key,
        kind="n",
        description=key,
        sense_rank=rank,
        tag_count=count,
        synset_id=key,
        ancestors=(
            SemanticAncestor("entity.n.01", 0, 5),
            SemanticAncestor(ancestor, 4, 1),
        ),
    )


class MeaningTests(unittest.TestCase):
    def test_rare_sense_can_support_taxonomy_clue_independently_of_vectors(self):
        words = ("BANK", "SHORE", "COAST", "EDGE")
        records = {
            word: WordKnowledge(
                word,
                vector,
                (sense(f"{word}-usual", f"unrelated-{word}", count=20),
                 sense(f"{word}-rare", "boundary.n.01", rank=5, count=0)),
            )
            for word, vector in zip(
                words, ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))
            )
        }

        batch = MeaningClueProvider(DictionaryCache(records)).find(words)
        by_relation = {item.relation: item for item in batch.evidence}

        self.assertEqual(by_relation["shared_hypernym"].strength, 4 / 6)
        self.assertEqual({match.source_id for match in by_relation["shared_hypernym"].matches},
                         {f"{word}-rare" for word in words})
        self.assertEqual({match.tag_count for match in by_relation["shared_hypernym"].matches}, {0})
        self.assertEqual(by_relation["surface_vector_similarity"].strength, 0.0)
        self.assertEqual(batch.coverage, (True,) * 4)

    def test_root_only_relation_is_not_a_clue(self):
        words = ("ONE", "TWO", "THREE", "FOUR")
        records = {
            word: WordKnowledge(
                word,
                None,
                (Interpretation("wordnet", word, "n", word, ancestors=(
                    SemanticAncestor("entity.n.01", 0, 1),
                )),),
            )
            for word in words
        }
        batch = MeaningClueProvider(DictionaryCache(records)).find(words)
        self.assertEqual(batch.evidence, ())

    def test_shared_synset_works_without_hypernyms(self):
        words = ("A", "B", "C", "D")
        shared = Interpretation(
            "wordnet", "shared-adjective", "a", "same meaning",
            synset_id="similar.a.01",
            ancestors=(SemanticAncestor("similar.a.01", 0, 0),),
        )
        records = {word: WordKnowledge(word, None, (shared,)) for word in words}

        batch = MeaningClueProvider(DictionaryCache(records)).find(words)

        self.assertEqual(batch.evidence[0].relation, "shared_synset")
        self.assertEqual(batch.evidence[0].strength, 1.0)

    def test_deletion_results_can_share_a_meaning(self):
        words = ("ACAT", "ABAT", "ARAT", "ADOG")
        edited = ("CAT", "BAT", "RAT", "DOG")
        records = {word: WordKnowledge(word, None) for word in words}
        records.update({word: WordKnowledge(word, None, (sense(word, "animal.n.01"),))
                        for word in edited})
        forms = ClueBatch(
            "word_form",
            (True,) * 4,
            (ClueEvidence(
                "word_form", "delete_one_letter", 4.0, 1.0,
                tuple(ClueMatch(word, derived) for word, derived in zip(words, edited)),
                "Each original word can lose one letter",
            ),),
        )

        provider = MeaningClueProvider(DictionaryCache(records))
        direct = provider.find(words)
        after_edits = provider.find_after_edits(words, forms, "delete_one_letter")

        self.assertEqual(direct.evidence, ())
        match = next(item for item in after_edits.evidence
                     if item.relation == "shared_hypernym_after_deletion")
        self.assertEqual(tuple(item.derived_word for item in match.matches), edited)
        self.assertEqual(match.strength, 4 / 6)


if __name__ == "__main__":
    unittest.main()
