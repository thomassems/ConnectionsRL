"""Inspectable clue-based player for exercising the environment before training."""

from __future__ import annotations

from dataclasses import dataclass

from .agent_input import ClueEvidence, FourWords
from .environment import GuessRecord, Observation
from .meanings import MeaningClueProvider
from .word_forms import WordFormClueProvider


@dataclass(frozen=True)
class CandidateScore:
    words: FourWords
    value: float
    evidence: tuple[ClueEvidence, ...]


class ClueBaseline:
    """Score legal guesses with fixed, deliberately simple clue weights.

    This is an evaluation baseline, not the trainable RL agent. Scores are
    useful for inspecting evidence but are not probabilities of correctness.
    """

    def __init__(
        self, meanings: MeaningClueProvider, forms: WordFormClueProvider
    ) -> None:
        self._meanings = meanings
        self._forms = forms

    def rank(
        self, observation: Observation, legal_guesses: tuple[FourWords, ...]
    ) -> tuple[CandidateScore, ...]:
        if observation.status != "in_progress":
            return ()
        scored = [
            self.score(guess, observation.guess_history)
            for guess in legal_guesses
            if not _ruled_out(guess, observation.guess_history)
        ]
        return tuple(sorted(scored, key=lambda item: (-item.value, tuple(sorted(item.words)))))

    def score(
        self, words: FourWords, history: tuple[GuessRecord, ...] = ()
    ) -> CandidateScore:
        direct = self._meanings.find(words)
        forms = self._forms.find(words)
        deleted = self._meanings.find_after_edits(words, forms, "delete_one_letter")
        added = self._meanings.find_after_edits(words, forms, "add_one_letter")
        evidence = direct.evidence + forms.evidence + deleted.evidence + added.evidence

        taxonomy = max(
            (item.strength for item in direct.evidence
             if item.relation in ("shared_synset", "shared_hypernym")),
            default=0.0,
        )
        vector = max(
            (item.strength for item in direct.evidence
             if item.relation == "surface_vector_similarity"),
            default=0.0,
        )
        edited = max(
            (item.strength for item in deleted.evidence + added.evidence),
            default=0.0,
        )
        affix = max(
            (item.strength for item in forms.evidence
             if item.relation in ("shared_prefix", "shared_suffix")),
            default=0.0,
        )
        one_away_hint = 0.05 if any(
            record.feedback == "one_away"
            and len(set(words).intersection(record.words)) == 3
            for record in history
        ) else 0.0
        value = max(taxonomy, 0.85 * edited) + 0.3 * vector + 0.2 * affix + one_away_hint
        return CandidateScore(words, value, evidence)


def _ruled_out(words: FourWords, history: tuple[GuessRecord, ...]) -> bool:
    """An 'incorrect' guess had at most two words from every unsolved group."""
    return any(
        record.feedback == "incorrect"
        and len(set(words).intersection(record.words)) >= 3
        for record in history
    )
