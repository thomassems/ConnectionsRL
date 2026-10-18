"""The compact vector cache preserves pretrained values and missing coverage."""

import gzip
import tempfile
import unittest
from pathlib import Path

from connections_rl import FastTextSubsetVectors
from scripts.build_fasttext_subset import build


class FastTextSubsetTests(unittest.TestCase):
    def test_build_and_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tiny.vec.gz"
            cache = Path(directory) / "tiny.sqlite3"
            with gzip.open(source, "wb") as stream:
                stream.write(b"3 2\ncat 1 0\ndog 0 1\nsun -1 0\n")

            metadata = build(source, cache, {"cat", "dog", "missing"})
            vectors = FastTextSubsetVectors(cache)
            try:
                self.assertEqual(metadata["matched_terms"], "2")
                self.assertEqual(vectors.vector("CAT"), (1.0, 0.0))
                self.assertEqual(vectors.vector("CAT DOG"), (0.5, 0.5))
                self.assertIsNone(vectors.vector("missing"))
                self.assertIsNone(vectors.vector("cat missing"))
                self.assertEqual(vectors.metadata["dimension"], "2")
            finally:
                vectors.close()


if __name__ == "__main__":
    unittest.main()
