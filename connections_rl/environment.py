"""A playable Connections episode for reinforcement learning agents."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

from .puzzle import Group, Puzzle
from .rewards import RewardConfig


Feedback = Literal["correct", "one_away", "incorrect"]
Status = Literal["in_progress", "won", "lost"]


@dataclass(frozen=True)
class GuessRecord:
    words: tuple[str, str, str, str]
    feedback: Feedback
    revealed_group: Group | None = None


@dataclass(frozen=True)
class Observation:
    puzzle_id: str
    remaining_words: tuple[str, ...]
    solved_groups: tuple[Group, ...]
    mistakes_left: int
    guess_history: tuple[GuessRecord, ...]
    status: Status


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    reward: float
    terminated: bool


class ConnectionsEnv:
    """Keep the answer key private and expose only game feedback.

    Call ``reset`` before ``step``. Invalid guesses raise ``ValueError`` and
    leave the episode unchanged. A completed episode must be reset before
    another guess can be submitted.
    """

    def __init__(self, puzzle: Puzzle, rewards: RewardConfig | None = None) -> None:
        self._puzzle = puzzle
        self.rewards = rewards if rewards is not None else RewardConfig()
        self._started = False

    def reset(self, seed: int | None = None) -> Observation:
        words = [word for group in self._puzzle.groups for word in group.words]
        random.Random(seed).shuffle(words)
        self._remaining_words = words
        self._solved_groups: list[Group] = []
        self._history: list[GuessRecord] = []
        self._guessed_combinations: set[frozenset[str]] = set()
        self._mistakes_left = 4
        self._status: Status = "in_progress"
        self._started = True
        return self.observation()

    def observation(self) -> Observation:
        self._require_started()
        return Observation(
            puzzle_id=self._puzzle.puzzle_id,
            remaining_words=tuple(self._remaining_words),
            solved_groups=tuple(self._solved_groups),
            mistakes_left=self._mistakes_left,
            guess_history=tuple(self._history),
            status=self._status,
        )

    def available_guesses(self) -> tuple[tuple[str, str, str, str], ...]:
        """Return all legal four-word actions in the current state."""
        self._require_started()
        if self._status != "in_progress":
            return ()
        return tuple(
            guess
            for guess in combinations(self._remaining_words, 4)
            if frozenset(guess) not in self._guessed_combinations
        )

    def step(self, guess: Sequence[str]) -> StepResult:
        self._require_started()
        if self._status != "in_progress":
            raise RuntimeError("episode is over; call reset before guessing again")

        words = self._validate_guess(guess)
        selection = frozenset(words)
        self._guessed_combinations.add(selection)

        for group in self._puzzle.groups:
            if group in self._solved_groups:
                continue
            if selection == frozenset(group.words):
                self._solved_groups.append(group)
                self._remaining_words = [
                    word for word in self._remaining_words if word not in selection
                ]
                self._history.append(GuessRecord(words, "correct", group))
                reward = self.rewards.correct_group
                if len(self._solved_groups) == 4:
                    self._status = "won"
                    reward += self.rewards.completion
                return StepResult(self.observation(), reward, self._status == "won")

        one_away = any(
            len(selection.intersection(group.words)) == 3
            for group in self._puzzle.groups
            if group not in self._solved_groups
        )
        self._mistakes_left -= 1
        feedback: Feedback = "one_away" if one_away else "incorrect"
        self._history.append(GuessRecord(words, feedback))
        if self._mistakes_left == 0:
            self._status = "lost"
            reward = self.rewards.failure
        else:
            reward = self.rewards.one_away if one_away else self.rewards.incorrect
        return StepResult(self.observation(), reward, self._status == "lost")

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("call reset before using the environment")

    def _validate_guess(self, guess: Sequence[str]) -> tuple[str, str, str, str]:
        if isinstance(guess, (str, bytes)) or not isinstance(guess, Sequence):
            raise ValueError("a guess must be a sequence of four words")
        if len(guess) != 4 or any(not isinstance(word, str) for word in guess):
            raise ValueError("a guess must contain four words")
        words = (guess[0], guess[1], guess[2], guess[3])
        if len(set(words)) != 4:
            raise ValueError("a guess must contain four distinct words")
        if not set(words).issubset(self._remaining_words):
            raise ValueError("a guess may use only remaining words")
        if frozenset(words) in self._guessed_combinations:
            raise ValueError("that four-word combination has already been guessed")
        return words
