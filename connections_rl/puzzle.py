"""Puzzle records and validation. Solutions remain private to the environment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


COLORS = frozenset({"yellow", "green", "blue", "purple"})


@dataclass(frozen=True)
class Group:
    title: str
    color: str
    words: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("group title must be a non-empty string")
        if not isinstance(self.color, str) or self.color not in COLORS:
            raise ValueError(f"group color must be one of {sorted(COLORS)}")
        if not isinstance(self.words, tuple):
            raise ValueError("group words must be a tuple")
        if len(self.words) != 4 or any(
            not isinstance(word, str) or not word.strip() for word in self.words
        ):
            raise ValueError("each group must contain four non-empty words")


@dataclass(frozen=True)
class Puzzle:
    puzzle_id: str
    groups: tuple[Group, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.puzzle_id, str) or not self.puzzle_id.strip():
            raise ValueError("puzzle_id must be a non-empty string")
        if not isinstance(self.groups, tuple) or any(
            not isinstance(group, Group) for group in self.groups
        ):
            raise ValueError("puzzle groups must be a tuple of Group objects")
        if len(self.groups) != 4:
            raise ValueError("a puzzle must contain exactly four groups")
        colors = [group.color for group in self.groups]
        if len(set(colors)) != 4:
            raise ValueError("a puzzle must contain one group of each color")
        words = [word.strip().casefold() for group in self.groups for word in group.words]
        if len(set(words)) != 16:
            raise ValueError("a puzzle must contain 16 distinct words")


def load_puzzle(path: str | Path) -> Puzzle:
    """Load a JSON puzzle into validated Python objects."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("groups"), list):
        raise ValueError("puzzle JSON must contain a groups list")
    try:
        if any(
            not isinstance(group, dict) or not isinstance(group.get("words"), list)
            for group in data["groups"]
        ):
            raise ValueError("each group must be an object with a words list")
        groups = tuple(
            Group(
                title=group["title"],
                color=group["color"],
                words=tuple(group["words"]),
            )
            for group in data["groups"]
        )
        return Puzzle(puzzle_id=data["puzzle_id"], groups=groups)
    except (KeyError, TypeError) as exc:
        raise ValueError("puzzle JSON has a missing or invalid field") from exc
