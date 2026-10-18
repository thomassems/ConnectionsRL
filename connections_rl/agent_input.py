"""Source-neutral knowledge and clue records for a future agent encoder.

These records preserve individual interpretations and clue evidence. A later
encoder can summarize them for the first network or pass individual evidence
items to a more flexible network without changing the knowledge sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


FourWords = tuple[str, str, str, str]
ClueChannel = Literal["meaning", "entity", "phrase", "word_form"]


@dataclass(frozen=True)
class SemanticAncestor:
    """A taxonomy node reachable from one sense, with its path distances."""

    source_id: str
    depth: int
    distance: int


@dataclass(frozen=True)
class Interpretation:
    """One possible meaning or entity match for a word.

    ``priority`` is a source-specific commonness hint, not a probability or a
    reason to discard this interpretation. ``None`` means unavailable.
    """

    source: str
    source_id: str
    kind: str
    description: str
    priority: float | None = None
    vector: tuple[float, ...] | None = None
    sense_rank: int | None = None
    tag_count: int | None = None
    synset_id: str | None = None
    ancestors: tuple[SemanticAncestor, ...] = ()

    def __post_init__(self) -> None:
        if self.vector is not None and not isinstance(self.vector, tuple):
            raise ValueError("interpretation vector must be a tuple")


@dataclass(frozen=True)
class WordKnowledge:
    """Cached, answer-independent information about one displayed word."""

    word: str
    base_vector: tuple[float, ...] | None
    interpretations: tuple[Interpretation, ...] = ()

    def __post_init__(self) -> None:
        if self.base_vector is not None and not isinstance(self.base_vector, tuple):
            raise ValueError("base vector must be a tuple")
        if not isinstance(self.interpretations, tuple):
            raise ValueError("interpretations must be a tuple")


class KnowledgeCache(Protocol):
    """Local lookup used before scoring; implementations may combine sources."""

    def lookup(self, word: str) -> WordKnowledge | None: ...


@dataclass(frozen=True)
class ClueMatch:
    """The word-level support behind a clue, with its original priority."""

    word: str
    source_id: str | None = None
    priority: float | None = None
    sense_rank: int | None = None
    tag_count: int | None = None
    derived_word: str | None = None


@dataclass(frozen=True)
class ClueEvidence:
    """One inspectable piece of group evidence from a clue provider.

    ``raw_score`` keeps the provider's own scale. ``strength`` is a provider-
    normalized value from 0 to 1; it is not assumed to be calibrated against
    other providers. Individual evidence stays available after summarizing.
    """

    channel: ClueChannel
    relation: str
    raw_score: float
    strength: float
    matches: tuple[ClueMatch, ...]
    explanation: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("clue strength must be between 0 and 1")
        if not isinstance(self.matches, tuple):
            raise ValueError("clue matches must be a tuple")


@dataclass(frozen=True)
class ClueBatch:
    """All evidence from one provider, including per-word source coverage."""

    channel: ClueChannel
    coverage: tuple[bool, bool, bool, bool]
    evidence: tuple[ClueEvidence, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.coverage, tuple) or len(self.coverage) != 4 or any(
            not isinstance(value, bool) for value in self.coverage
        ):
            raise ValueError("clue coverage must have four boolean values")
        if not isinstance(self.evidence, tuple):
            raise ValueError("clue evidence must be a tuple")
        if any(item.channel != self.channel for item in self.evidence):
            raise ValueError("clue evidence must match the batch channel")


@dataclass(frozen=True)
class ChannelSummary:
    """Fixed-size candidate features for one clue channel.

    ``specific`` follows the caller's ordered relation names, so a model can
    use a stable feature order while the full evidence remains in ClueBatch.
    """

    channel: ClueChannel
    coverage_count: int
    evidence_count: int
    best_strength: float
    second_strength: float
    best_support_count: int
    best_priority_mean: float
    best_priority_known: bool
    best_sense_rank_mean: float
    best_sense_rank_known: bool
    best_tag_count_mean: float
    best_tag_count_known: bool
    specific: tuple[tuple[str, float], ...]


def summarize_clues(
    batch: ClueBatch, relation_names: tuple[str, ...] = ()
) -> ChannelSummary:
    """Summarize evidence without discarding rare interpretations from cache."""
    ordered = sorted(batch.evidence, key=lambda item: item.strength, reverse=True)
    best = ordered[0] if ordered else None
    known_priorities = (
        tuple(match.priority for match in best.matches if match.priority is not None)
        if best is not None
        else ()
    )
    known_ranks = (
        tuple(match.sense_rank for match in best.matches if match.sense_rank is not None)
        if best is not None
        else ()
    )
    known_counts = (
        tuple(match.tag_count for match in best.matches if match.tag_count is not None)
        if best is not None
        else ()
    )
    return ChannelSummary(
        channel=batch.channel,
        coverage_count=sum(batch.coverage),
        evidence_count=len(ordered),
        best_strength=best.strength if best is not None else 0.0,
        second_strength=ordered[1].strength if len(ordered) > 1 else 0.0,
        best_support_count=len({match.word for match in best.matches}) if best else 0,
        best_priority_mean=(
            sum(known_priorities) / len(known_priorities) if known_priorities else 0.0
        ),
        best_priority_known=bool(known_priorities),
        best_sense_rank_mean=sum(known_ranks) / len(known_ranks) if known_ranks else 0.0,
        best_sense_rank_known=bool(known_ranks),
        best_tag_count_mean=sum(known_counts) / len(known_counts) if known_counts else 0.0,
        best_tag_count_known=bool(known_counts),
        specific=tuple(
            (
                name,
                max(
                    (item.strength for item in ordered if item.relation == name),
                    default=0.0,
                ),
            )
            for name in relation_names
        ),
    )


class ClueProvider(Protocol):
    """Produce evidence for a proposed group using cached prior knowledge."""

    channel: ClueChannel

    def find(self, words: FourWords) -> ClueBatch: ...
