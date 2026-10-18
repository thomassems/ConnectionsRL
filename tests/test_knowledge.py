"""Local knowledge adapters work without downloading models in the test suite."""

import unittest

from connections_rl import FastTextVectors, WordFormClueProvider, WordNetKnowledgeCache


class FakeFastText:
    def get_word_vector(self, word):
        return {"bank": (1.0, 0.0), "river": (0.0, 1.0), "money": (1.0, 1.0)}.get(
            word, (0.0, 0.0)
        )


class FakeNode:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name


class FakeSynset:
    def __init__(self, definition):
        self._definition = definition

    def definition(self):
        return self._definition

    def name(self):
        return self._definition + ".n.01"

    def hypernym_paths(self):
        return [[FakeNode("entity.n.01"), FakeNode("boundary.n.01"), self]]


class FakeLemma:
    def __init__(self, key, definition, count):
        self._key = key
        self._synset = FakeSynset(definition)
        self._count = count

    def key(self):
        return self._key

    def synset(self):
        return self._synset

    def count(self):
        return self._count


class FakeWordNet:
    def __init__(self):
        self.calls = 0

    def lemmas(self, word, pos=None):
        self.calls += 1
        if word == "bank" and pos == "n":
            return (
                FakeLemma("financial-bank", "money", 18),
                FakeLemma("river-bank", "river", 0),
            )
        return ()


class KnowledgeTests(unittest.TestCase):
    def test_wordnet_retains_rare_senses_and_caches_lookup(self):
        reader = FakeWordNet()
        cache = WordNetKnowledgeCache(reader, FastTextVectors(FakeFastText()))

        record = cache.lookup("BANK")
        calls_after_first_lookup = reader.calls

        self.assertIs(record, cache.lookup("BANK"))
        self.assertEqual(reader.calls, calls_after_first_lookup)
        self.assertEqual(record.base_vector, (1.0, 0.0))
        self.assertEqual([sense.source_id for sense in record.interpretations], ["financial-bank", "river-bank"])
        self.assertEqual([sense.sense_rank for sense in record.interpretations], [1, 2])
        self.assertEqual([sense.tag_count for sense in record.interpretations], [18, 0])
        self.assertEqual(record.interpretations[1].vector, (0.0, 1.0))
        self.assertEqual(
            [(node.source_id, node.depth, node.distance)
             for node in record.interpretations[1].ancestors],
            [("boundary.n.01", 1, 1), ("entity.n.01", 0, 2), ("river.n.01", 2, 0)],
        )

    def test_fasttext_phrase_vector_averages_token_vectors(self):
        vectors = FastTextVectors(FakeFastText())
        self.assertEqual(vectors.vector("BANK RIVER"), (0.5, 0.5))

    def test_word_form_evidence_keeps_individual_edit_results(self):
        lexicon = {"at", "cat", "bat", "rat", "mat", "scat", "brat"}
        provider = WordFormClueProvider(lambda word: word in lexicon)

        batch = provider.find(("CAT", "BAT", "RAT", "MAT"))
        relations = {item.relation: item for item in batch.evidence}

        self.assertEqual(relations["shared_suffix"].matches[0].source_id, "at")
        self.assertEqual(relations["delete_one_letter"].strength, 1.0)
        self.assertEqual(
            {match.source_id for match in relations["delete_one_letter"].matches},
            {"at"},
        )
        self.assertIn("scat", {match.source_id for match in relations["add_one_letter"].matches})
        self.assertEqual(batch.coverage, (True,) * 4)


if __name__ == "__main__":
    unittest.main()
