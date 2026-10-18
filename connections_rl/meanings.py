"""Inspectable meaning clues from WordNet relations and fixed vectors."""

from __future__ import annotations

from math import sqrt

from .agent_input import (
    ClueBatch,
    ClueEvidence,
    ClueMatch,
    FourWords,
    Interpretation,
    KnowledgeCache,
    SemanticAncestor,
    WordKnowledge,
)


class MeaningClueProvider:
    """Compare senses through taxonomy and surface words through vector geometry.

    A high clue strength is evidence, not a probability that a group is correct.
    All senses remain available in the knowledge cache; no commonness cutoff is
    applied when finding a shared WordNet ancestor.
    """

    channel = "meaning"

    def __init__(self, knowledge: KnowledgeCache) -> None:
        self._knowledge = knowledge
        self._pair_cosine: dict[tuple[str, str], float | None] = {}

    def find(self, words: FourWords) -> ClueBatch:
        _validate(words)
        records = tuple(self._knowledge.lookup(word) for word in words)
        coverage = tuple(record is not None for record in records)
        if any(record is None for record in records):
            return ClueBatch(self.channel, coverage, ())
        choices = tuple(((word, record),) for word, record in zip(words, records))
        evidence = list(self._taxonomy_evidence(words, choices))
        vectors = tuple(record.base_vector for record in records)
        if all(vector is not None for vector in vectors):
            similarities = []
            for left in range(4):
                for right in range(left + 1, 4):
                    key = tuple(sorted((words[left], words[right])))
                    if key not in self._pair_cosine:
                        self._pair_cosine[key] = _cosine(vectors[left], vectors[right])
                    similarities.append(self._pair_cosine[key])
            vector_evidence = _vector_evidence(words, tuple(similarities))
            if vector_evidence is not None:
                evidence.append(vector_evidence)
        return ClueBatch(self.channel, coverage, tuple(evidence))

    def find_after_edits(
        self, words: FourWords, forms: ClueBatch, relation: str
    ) -> ClueBatch:
        """Check whether valid spelling edits share a WordNet ancestor."""
        _validate(words)
        if forms.channel != "word_form" or relation not in (
            "delete_one_letter", "add_one_letter"
        ):
            raise ValueError("expected a one-letter word-form clue")
        edited = {
            word: {match.source_id for item in forms.evidence if item.relation == relation
                   for match in item.matches if match.word == word and match.source_id}
            for word in words
        }
        coverage = tuple(bool(edited[word]) for word in words)
        if not all(coverage):
            return ClueBatch(self.channel, coverage, ())
        choices = tuple(
            tuple((variant, record) for variant in sorted(edited[word])
                  if (record := self._knowledge.lookup(variant)) is not None)
            for word in words
        )
        if any(not option for option in choices):
            return ClueBatch(self.channel, tuple(bool(option) for option in choices), ())
        suffix = "after_deletion" if relation == "delete_one_letter" else "after_addition"
        return ClueBatch(
            self.channel,
            coverage,
            tuple(self._taxonomy_evidence(words, choices, suffix)),
        )

    def _taxonomy_evidence(
        self,
        words: FourWords,
        choices: tuple[tuple[tuple[str, WordKnowledge], ...], ...],
        suffix: str = "",
    ) -> list[ClueEvidence]:
        # One index per original word. Each index retains every sense of every
        # candidate spelling, so a rare interpretation can supply the link.
        indexes: list[dict[str, list[tuple[str, Interpretation, SemanticAncestor]]]] = []
        for options in choices:
            index: dict[str, list[tuple[str, Interpretation, SemanticAncestor]]] = {}
            for derived_word, record in options:
                for sense in record.interpretations:
                    for ancestor in sense.ancestors:
                        if ancestor.depth > 0 or ancestor.source_id == sense.synset_id:
                            index.setdefault(ancestor.source_id, []).append(
                                (derived_word, sense, ancestor)
                            )
            indexes.append(index)
        if not indexes:
            return []
        shared = set.intersection(*(set(index) for index in indexes))
        evidence = []
        for ancestor_id in shared:
            selected = [
                min(
                    index[ancestor_id],
                    key=lambda option: (
                        option[2].distance,
                        -option[2].depth,
                        option[1].sense_rank or 10**9,
                    ),
                )
                for index in indexes
            ]
            depth = min(option[2].depth for option in selected)
            mean_distance = sum(option[2].distance for option in selected) / 4
            exact = all(option[1].synset_id == ancestor_id for option in selected)
            relation = "shared_synset" if exact else "shared_hypernym"
            if suffix:
                relation += f"_{suffix}"
            strength = 1.0 if exact else depth / (depth + mean_distance + 1)
            matches = tuple(
                ClueMatch(
                    word=word,
                    source_id=sense.source_id,
                    priority=sense.priority,
                    sense_rank=sense.sense_rank,
                    tag_count=sense.tag_count,
                    derived_word=derived if derived != word else None,
                )
                for word, (derived, sense, _) in zip(words, selected)
            )
            evidence.append(
                ClueEvidence(
                    channel=self.channel,
                    relation=relation,
                    raw_score=float(depth),
                    strength=strength,
                    matches=matches,
                    explanation=f"Selected senses share WordNet node {ancestor_id}",
                )
            )
        return sorted(evidence, key=lambda item: (-item.strength, item.explanation))


def _vector_evidence(
    words: FourWords, similarities: tuple[float | None, ...]
) -> ClueEvidence | None:
    if any(value is None for value in similarities):
        return None
    values = tuple(value for value in similarities if value is not None)
    return ClueEvidence(
        channel="meaning",
        relation="surface_vector_similarity",
        raw_score=sum(values) / len(values),
        strength=max(0.0, min(values)),
        matches=tuple(ClueMatch(word) for word in words),
        explanation="Cosine similarity of all six pairs of fixed surface-word vectors",
    )


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float | None:
    if len(left) != len(right):
        raise ValueError("word vectors must have one consistent dimension")
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return None
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)))


def _validate(words: FourWords) -> None:
    if len(words) != 4 or any(not isinstance(word, str) or not word for word in words):
        raise ValueError("a clue candidate must contain four non-empty words")
