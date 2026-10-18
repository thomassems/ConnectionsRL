"""Small candidate-value network and answer-independent puzzle features."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .agent_input import ChannelSummary, summarize_clues
from .environment import Observation
from .knowledge import VectorSource, WordNetKnowledgeCache
from .meanings import MeaningClueProvider
from .puzzle import Puzzle
from .word_forms import WordFormClueProvider


FEATURE_VERSION = 1
HISTORY_SLOTS = 7
VECTOR_DIM = 300
MEANING_RELATIONS = ("shared_synset", "shared_hypernym", "surface_vector_similarity")
FORM_RELATIONS = ("shared_prefix", "shared_suffix", "delete_one_letter", "add_one_letter")
CLUE_DIM = 2 * 11 + len(MEANING_RELATIONS) + len(FORM_RELATIONS) + 2  # empty entity/phrase masks
STATE_DIM = HISTORY_SLOTS * (2 * VECTOR_DIM + 8) + 2


def _summary_values(summary: ChannelSummary) -> list[float]:
    return [
        summary.coverage_count / 4,
        min(summary.evidence_count, 20) / 20,
        summary.best_strength,
        summary.second_strength,
        summary.best_support_count / 4,
        summary.best_priority_mean,
        float(summary.best_priority_known),
        1 / (1 + summary.best_sense_rank_mean) if summary.best_sense_rank_known else 0,
        float(summary.best_sense_rank_known),
        min(summary.best_tag_count_mean, 50) / 50,
        float(summary.best_tag_count_known),
        *(strength for _, strength in summary.specific),
    ]


@dataclass(frozen=True)
class PuzzleFeatures:
    words: tuple[str, ...]
    word_vectors: np.ndarray  # 16 x 300; fixed pretrained vectors
    candidates: np.ndarray  # 1820 x 4 word indexes
    clues: np.ndarray  # 1820 x CLUE_DIM

    def legal_indices(self, observation: Observation) -> np.ndarray:
        if observation.status != "in_progress":
            return np.empty(0, dtype=np.int64)
        remaining = {self.words.index(word) for word in observation.remaining_words}
        previously_guessed = {
            frozenset(self.words.index(word) for word in record.words)
            for record in observation.guess_history
        }
        return np.asarray([
            index for index, candidate in enumerate(self.candidates)
            if set(candidate).issubset(remaining)
            and frozenset(candidate) not in previously_guessed
        ], dtype=np.int64)

    def guess(self, index: int) -> tuple[str, str, str, str]:
        return tuple(self.words[int(i)] for i in self.candidates[index])  # type: ignore[return-value]

    def overlaps(self, observation: Observation, indices: np.ndarray) -> np.ndarray:
        result = np.zeros((len(indices), HISTORY_SLOTS), dtype=np.float32)
        for slot, record in enumerate(observation.guess_history[:HISTORY_SLOTS]):
            previous = {self.words.index(word) for word in record.words}
            result[:, slot] = np.isin(self.candidates[indices], list(previous)).sum(axis=1) / 4
        return result

    def state(self, observation: Observation, vectors: VectorSource) -> np.ndarray:
        state = np.zeros(STATE_DIM, dtype=np.float32)
        state[0] = observation.mistakes_left / 4
        state[1] = len(observation.remaining_words) / 16
        for slot, record in enumerate(observation.guess_history[:HISTORY_SLOTS]):
            offset = 2 + slot * (2 * VECTOR_DIM + 8)
            word_ids = [self.words.index(word) for word in record.words]
            state[offset:offset + VECTOR_DIM] = self.word_vectors[word_ids].mean(axis=0)
            if record.revealed_group is not None:
                title = vectors.vector(record.revealed_group.title)
                if title is not None:
                    state[offset + VECTOR_DIM:offset + 2 * VECTOR_DIM] = title
            feedback = {"correct": 0, "one_away": 1, "incorrect": 2}[record.feedback]
            state[offset + 2 * VECTOR_DIM + feedback] = 1
            if record.revealed_group is not None:
                color = ("yellow", "green", "blue", "purple").index(record.revealed_group.color)
                state[offset + 2 * VECTOR_DIM + 3 + color] = 1
            state[offset + 2 * VECTOR_DIM + 7] = 1
        return state


def build_features(
    puzzle: Puzzle, knowledge: WordNetKnowledgeCache,
    meanings: MeaningClueProvider, forms: WordFormClueProvider,
) -> PuzzleFeatures:
    # Puzzle answers are used only to obtain the 16 displayed words. They are
    # never included in the feature rows or model state before game feedback.
    words = tuple(sorted(word for group in puzzle.groups for word in group.words))
    vectors = np.asarray([
        knowledge.lookup(word).base_vector or (0.0,) * VECTOR_DIM for word in words
    ], dtype=np.float32)
    if vectors.shape != (16, VECTOR_DIM):
        raise ValueError("network requires 300-dimensional word vectors")
    candidates = np.asarray(list(combinations(range(16), 4)), dtype=np.int16)
    clues = np.zeros((len(candidates), CLUE_DIM), dtype=np.float32)
    for index, row in enumerate(candidates):
        guess = tuple(words[int(i)] for i in row)
        meaning = summarize_clues(meanings.find(guess), MEANING_RELATIONS)
        form = summarize_clues(forms.find(guess), FORM_RELATIONS)
        values = _summary_values(meaning) + [0.0] + [0.0] + _summary_values(form)
        clues[index] = values
    return PuzzleFeatures(words, vectors, candidates, clues)


def load_or_build_features(
    path: Path, cache_dir: Path, knowledge: WordNetKnowledgeCache,
    meanings: MeaningClueProvider, forms: WordFormClueProvider,
) -> PuzzleFeatures:
    from .puzzle import load_puzzle

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{path.stem}.npz"
    if target.exists():
        with np.load(target, allow_pickle=False) as item:
            if (int(item["version"]) == FEATURE_VERSION
                    and str(item["puzzle_sha256"]) == digest
                    and item["clues"].shape[1] == CLUE_DIM):
                return PuzzleFeatures(
                    tuple(json.loads(str(item["words"]))), item["word_vectors"],
                    item["candidates"], item["clues"],
                )
    features = build_features(load_puzzle(path), knowledge, meanings, forms)
    np.savez_compressed(
        target, version=FEATURE_VERSION, puzzle_sha256=digest,
        words=json.dumps(features.words), word_vectors=features.word_vectors,
        candidates=features.candidates, clues=features.clues,
    )
    return features


class CandidateQNetwork(nn.Module):
    """Shared, permutation-invariant group scorer with an ordered history input."""

    def __init__(self, clue_dim: int = CLUE_DIM, hidden: int = 64) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(nn.Linear(STATE_DIM, hidden), nn.ReLU())
        self.group_encoder = nn.Sequential(
            nn.Linear(2 * VECTOR_DIM + clue_dim + HISTORY_SLOTS, hidden), nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(
        self, word_vectors: torch.Tensor, candidate_ids: torch.Tensor,
        clues: torch.Tensor, state: torch.Tensor, overlaps: torch.Tensor,
    ) -> torch.Tensor:
        candidate_words = word_vectors[candidate_ids.long()]
        pooled = torch.cat((candidate_words.mean(dim=1), candidate_words.std(dim=1, unbiased=False)), dim=1)
        group = self.group_encoder(torch.cat((pooled, clues, overlaps), dim=1))
        context = self.state_encoder(state.reshape(1, -1)).expand(len(group), -1)
        return self.value(torch.cat((group, context), dim=1)).squeeze(1)


def score_candidates(
    model: CandidateQNetwork, features: PuzzleFeatures, observation: Observation,
    indices: np.ndarray, vectors: VectorSource,
) -> torch.Tensor:
    return model(
        torch.from_numpy(features.word_vectors),
        torch.from_numpy(features.candidates[indices].astype(np.int64)),
        torch.from_numpy(features.clues[indices]),
        torch.from_numpy(features.state(observation, vectors)),
        torch.from_numpy(features.overlaps(observation, indices)),
    )
