"""Run an inspectable baseline episode: python -m connections_rl.demo."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .baseline import CandidateScore, ClueBaseline
from .environment import ConnectionsEnv
from .knowledge import FastTextSubsetVectors, FastTextVectors, VectorSource, WordNetKnowledgeCache
from .meanings import MeaningClueProvider
from .puzzle import load_puzzle
from .word_forms import WordFormClueProvider


DEFAULT_PUZZLE = Path(__file__).resolve().parent.parent / "data/puzzles/example.json"


class NoVectors:
    """Run WordNet-only when no local fastText binary is available."""

    def vector(self, word: str) -> None:
        return None


@dataclass(frozen=True)
class EpisodeReport:
    status: str
    turns: int
    total_reward: float


def play(
    env: ConnectionsEnv, baseline: ClueBaseline, seed: int = 7,
    show_top: int = 3
) -> EpisodeReport:
    observation = env.reset(seed=seed)
    total_reward = 0.0
    turns = 0
    while observation.status == "in_progress":
        ranked = baseline.rank(observation, env.available_guesses())
        if not ranked:
            raise RuntimeError("no candidate guesses remain")
        print(f"\nTurn {turns + 1} | mistakes left: {observation.mistakes_left}")
        for index, candidate in enumerate(ranked[:show_top], 1):
            print(f"  {index}. {', '.join(candidate.words)} | clue score {candidate.value:.3f}")
            _print_evidence(candidate)
        choice = ranked[0]
        result = env.step(choice.words)
        observation = result.observation
        total_reward += result.reward
        turns += 1
        feedback = observation.guess_history[-1].feedback
        print(f"  Guess: {', '.join(choice.words)} -> {feedback}; reward {result.reward:+g}")
        revealed = observation.guess_history[-1].revealed_group
        if revealed is not None:
            print(f"  Revealed: {revealed.title} ({revealed.color})")
    print(f"\nEpisode: {observation.status} in {turns} turns; total reward {total_reward:+g}")
    return EpisodeReport(observation.status, turns, total_reward)


def _print_evidence(candidate: CandidateScore) -> None:
    ordered = sorted(
        (item for item in candidate.evidence
         if item.relation not in ("delete_one_letter", "add_one_letter")),
        key=lambda item: (-item.strength, item.relation)
    )
    strongest = []
    seen_relations = set()
    for item in ordered:
        if item.relation not in seen_relations:
            strongest.append(item)
            seen_relations.add(item.relation)
        if len(strongest) == 2:
            break
    for item in strongest:
        print(f"     {item.relation}: {item.strength:.3f} ({item.explanation})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puzzle", type=Path, default=DEFAULT_PUZZLE)
    parser.add_argument("--fasttext", type=Path, help="local pretrained fastText .bin model")
    parser.add_argument("--vector-cache", type=Path,
                        help="compact SQLite cache of pretrained fastText vectors")
    parser.add_argument("--wordnet-data", type=Path, help="directory containing NLTK corpora")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--show-top", type=int, default=3)
    args = parser.parse_args(argv)
    if args.show_top < 1:
        parser.error("--show-top must be positive")
    if args.fasttext and args.vector_cache:
        parser.error("choose either --fasttext or --vector-cache")

    try:
        import nltk
    except ImportError:
        parser.error("NLTK is required; install it with pip install -e '.[wordnet]'")
    local_data = Path(__file__).resolve().parent.parent / ".venv/nltk_data"
    data_dir = args.wordnet_data or (local_data if local_data.exists() else None)
    if data_dir is not None:
        if not data_dir.is_dir():
            parser.error(f"WordNet data directory does not exist: {data_dir}")
        nltk.data.path.insert(0, str(data_dir.resolve()))

    try:
        vectors: VectorSource = (
            FastTextVectors.from_file(args.fasttext) if args.fasttext
            else FastTextSubsetVectors(args.vector_cache) if args.vector_cache
            else NoVectors()
        )
    except (RuntimeError, OSError, ValueError) as exc:
        parser.error(str(exc))
    knowledge = WordNetKnowledgeCache.from_nltk(vectors)
    baseline = ClueBaseline(
        MeaningClueProvider(knowledge), WordFormClueProvider(knowledge.contains)
    )
    try:
        play(ConnectionsEnv(load_puzzle(args.puzzle)), baseline,
             seed=args.seed, show_top=args.show_top)
    except RuntimeError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
