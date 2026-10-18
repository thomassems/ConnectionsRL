"""Answer-independent spelling clues for proposed four-word groups."""

from __future__ import annotations

from collections.abc import Callable
from string import ascii_lowercase

from .agent_input import ClueBatch, ClueEvidence, ClueMatch, FourWords


class WordFormClueProvider:
    """Expose shared affixes and valid one-letter edits as inspectable clues.

    ``is_word`` should query a local lexicon, such as WordNetKnowledgeCache.contains.
    Edit results are kept individually; their shared meaning is a separate clue.
    """

    channel = "word_form"

    def __init__(self, is_word: Callable[[str], bool]) -> None:
        self._is_word = is_word
        self._edits: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}

    def find(self, words: FourWords) -> ClueBatch:
        if len(words) != 4 or any(not isinstance(word, str) or not word for word in words):
            raise ValueError("a clue candidate must contain four non-empty words")
        normalized = tuple(word.casefold() for word in words)
        evidence: list[ClueEvidence] = []

        for relation, affix in (
            ("shared_prefix", _common_prefix(normalized)),
            ("shared_suffix", _common_suffix(normalized)),
        ):
            if affix:
                shortest = min(len(word) for word in normalized)
                evidence.append(
                    ClueEvidence(
                        channel=self.channel,
                        relation=relation,
                        raw_score=float(len(affix)),
                        strength=len(affix) / shortest,
                        matches=tuple(ClueMatch(word, affix) for word in words),
                        explanation=f"All four words share {relation.removeprefix('shared_')} {affix!r}",
                    )
                )

        for relation, edit_index in (("delete_one_letter", 0), ("add_one_letter", 1)):
            matches = tuple(
                ClueMatch(word, edited)
                for word in words
                for edited in self._valid_edits(word.casefold())[edit_index]
            )
            if matches:
                support = len({match.word for match in matches})
                evidence.append(
                    ClueEvidence(
                        channel=self.channel,
                        relation=relation,
                        raw_score=float(support),
                        strength=support / 4,
                        matches=matches,
                        explanation=f"{support} candidate words have valid {relation} results",
                    )
                )

        return ClueBatch(self.channel, (True, True, True, True), tuple(evidence))

    def _valid_edits(self, word: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if word not in self._edits:
            if not word.isalpha():
                self._edits[word] = ((), ())
            else:
                deletions = {word[:index] + word[index + 1 :] for index in range(len(word))}
                additions = {
                    word[:index] + letter + word[index:]
                    for index in range(len(word) + 1)
                    for letter in ascii_lowercase
                }
                valid_deletions = tuple(
                    sorted(candidate for candidate in deletions if candidate and self._is_word(candidate))
                )
                valid_additions = tuple(
                    sorted(candidate for candidate in additions if self._is_word(candidate))
                )
                self._edits[word] = (valid_deletions, valid_additions)
        return self._edits[word]


def _common_prefix(words: tuple[str, ...]) -> str:
    return _common_prefix_pairwise(words)


def _common_suffix(words: tuple[str, ...]) -> str:
    return _common_prefix_pairwise(tuple(word[::-1] for word in words))[::-1]


def _common_prefix_pairwise(words: tuple[str, ...]) -> str:
    prefix = words[0]
    for word in words[1:]:
        matching = 0
        for left, right in zip(prefix, word):
            if left != right:
                break
            matching += 1
        prefix = prefix[:matching]
    return prefix
