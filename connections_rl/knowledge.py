"""Adapters for fixed fastText vectors and cached WordNet interpretations."""

from __future__ import annotations

import re
import sqlite3
import struct
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from .agent_input import Interpretation, SemanticAncestor, WordKnowledge


class VectorSource(Protocol):
    def vector(self, word: str) -> tuple[float, ...] | None: ...


class FastTextVectors:
    """Read vectors from an already loaded fastText binary model."""

    def __init__(self, model: object) -> None:
        self._model = model

    @classmethod
    def from_file(cls, path: str | Path) -> FastTextVectors:
        try:
            import fasttext
        except ImportError as exc:
            raise RuntimeError("install fasttext to load a .bin model") from exc
        return cls(fasttext.load_model(str(path)))

    def vector(self, word: str) -> tuple[float, ...] | None:
        tokens = word.casefold().split()
        if not tokens:
            return None
        vectors = [
            tuple(float(value) for value in self._model.get_word_vector(token))
            for token in tokens
        ]
        return _average(vectors)


class FastTextSubsetVectors:
    """Read a compact SQLite cache of official pretrained fastText text vectors.

    Unlike the full fastText binary, this source cannot synthesize vectors for
    out-of-vocabulary words. Missing words return ``None`` explicitly.
    """

    def __init__(self, path: str | Path) -> None:
        database = Path(path).resolve()
        self._connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
        row = self._connection.execute(
            "SELECT value FROM metadata WHERE key = 'dimension'"
        ).fetchone()
        if row is None:
            self._connection.close()
            raise ValueError("fastText cache has no vector dimension")
        self.dimension = int(row[0])
        self.metadata = dict(self._connection.execute("SELECT key, value FROM metadata"))

    @lru_cache(maxsize=65536)
    def _token_vector(self, token: str) -> tuple[float, ...] | None:
        row = self._connection.execute(
            "SELECT data FROM vectors WHERE word = ?", (token,)
        ).fetchone()
        if row is None:
            return None
        if len(row[0]) != 4 * self.dimension:
            raise ValueError(f"invalid cached vector size for {token!r}")
        return struct.unpack(f"<{self.dimension}f", row[0])

    def vector(self, word: str) -> tuple[float, ...] | None:
        tokens = word.casefold().split()
        if not tokens:
            return None
        vectors = [self._token_vector(token) for token in tokens]
        if any(vector is None for vector in vectors):
            return None
        return _average(vectors)

    def close(self) -> None:
        self._token_vector.cache_clear()
        self._connection.close()


class WordNetKnowledgeCache:
    """Retain every WordNet lemma sense, including rarely tagged senses."""

    POS = ("n", "v", "a", "s", "r")

    def __init__(self, reader: object, vectors: VectorSource) -> None:
        self._reader = reader
        self._vectors = vectors
        self._cache: dict[str, WordKnowledge] = {}
        self._known_words: dict[str, bool] = {}

    @classmethod
    def from_nltk(cls, vectors: VectorSource) -> WordNetKnowledgeCache:
        try:
            from nltk.corpus import wordnet
        except ImportError as exc:
            raise RuntimeError("install nltk and its wordnet corpus") from exc
        return cls(wordnet, vectors)

    def lookup(self, word: str) -> WordKnowledge:
        if word in self._cache:
            return self._cache[word]
        normalized = word.casefold().replace(" ", "_")
        interpretations: list[Interpretation] = []
        try:
            for pos in self.POS:
                for rank, lemma in enumerate(self._reader.lemmas(normalized, pos=pos), 1):
                    synset = lemma.synset()
                    definition = synset.definition()
                    definition_words = re.findall(r"[a-z]+(?:'[a-z]+)?", definition.casefold())
                    definition_vectors = [
                        vector
                        for token in definition_words
                        if (vector := self._vectors.vector(token)) is not None
                    ]
                    interpretations.append(
                        Interpretation(
                            source="wordnet",
                            source_id=lemma.key(),
                            kind=pos,
                            description=definition,
                            vector=_average(definition_vectors),
                            sense_rank=rank,
                            tag_count=lemma.count(),
                            synset_id=synset.name(),
                            ancestors=_ancestors(synset.hypernym_paths()),
                        )
                    )
        except LookupError as exc:
            raise RuntimeError("download the nltk wordnet corpus before lookup") from exc
        record = WordKnowledge(word, self._vectors.vector(word), tuple(interpretations))
        self._cache[word] = record
        return record

    def contains(self, word: str) -> bool:
        """Check a transformed word against WordNet without building vectors."""
        normalized = word.casefold().replace(" ", "_")
        if normalized not in self._known_words:
            try:
                self._known_words[normalized] = bool(self._reader.lemmas(normalized))
            except LookupError as exc:
                raise RuntimeError("download the nltk wordnet corpus before lookup") from exc
        return self._known_words[normalized]


def _average(vectors: list[tuple[float, ...]]) -> tuple[float, ...] | None:
    if not vectors:
        return None
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("word vectors must have one consistent dimension")
    return tuple(sum(vector[index] for vector in vectors) / len(vectors) for index in range(dimension))


def _ancestors(paths: list[list[object]]) -> tuple[SemanticAncestor, ...]:
    """Collapse multiple WordNet paths, retaining the closest route to each node."""
    by_id: dict[str, SemanticAncestor] = {}
    for path in paths:
        for depth, node in enumerate(path):
            source_id = node.name()
            candidate = SemanticAncestor(source_id, depth, len(path) - depth - 1)
            current = by_id.get(source_id)
            if current is None or (candidate.distance, -candidate.depth) < (
                current.distance,
                -current.depth,
            ):
                by_id[source_id] = candidate
    return tuple(sorted(by_id.values(), key=lambda item: item.source_id))
