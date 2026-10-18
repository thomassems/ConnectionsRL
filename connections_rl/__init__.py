"""Core types for the Connections RL project."""

from .agent_input import (
    ChannelSummary,
    ClueBatch,
    ClueEvidence,
    ClueMatch,
    ClueProvider,
    Interpretation,
    KnowledgeCache,
    SemanticAncestor,
    WordKnowledge,
    summarize_clues,
)
from .environment import ConnectionsEnv, GuessRecord, Observation, StepResult
from .knowledge import FastTextSubsetVectors, FastTextVectors, VectorSource, WordNetKnowledgeCache
from .meanings import MeaningClueProvider
from .puzzle import Group, Puzzle, load_puzzle
from .rewards import RewardConfig
from .word_forms import WordFormClueProvider

__all__ = [
    "ConnectionsEnv",
    "ChannelSummary",
    "ClueBatch",
    "ClueEvidence",
    "ClueMatch",
    "ClueProvider",
    "FastTextVectors",
    "FastTextSubsetVectors",
    "Group",
    "GuessRecord",
    "Interpretation",
    "KnowledgeCache",
    "MeaningClueProvider",
    "Observation",
    "Puzzle",
    "RewardConfig",
    "SemanticAncestor",
    "StepResult",
    "VectorSource",
    "WordKnowledge",
    "WordFormClueProvider",
    "WordNetKnowledgeCache",
    "load_puzzle",
    "summarize_clues",
]
