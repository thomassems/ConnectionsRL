"""Evaluate the clue baseline over a directory of saved puzzle JSON files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median

from .baseline import ClueBaseline
from .demo import NoVectors
from .environment import ConnectionsEnv
from .knowledge import FastTextSubsetVectors, FastTextVectors, VectorSource, WordNetKnowledgeCache
from .meanings import MeaningClueProvider
from .puzzle import load_puzzle
from .word_forms import WordFormClueProvider


def evaluate_one(path: Path, baseline: ClueBaseline, seed: int) -> dict:
    puzzle = load_puzzle(path)
    env = ConnectionsEnv(puzzle)
    observation = env.reset(seed=seed)
    total_reward = 0.0
    guesses = []
    initial_true_group_ranks = {}
    while observation.status == "in_progress":
        ranked = baseline.rank(observation, env.available_guesses())
        if not ranked:
            raise RuntimeError(f"no legal candidate remains for {puzzle.puzzle_id}")
        if not guesses:
            ranks = {frozenset(candidate.words): index
                     for index, candidate in enumerate(ranked, 1)}
            initial_true_group_ranks = {
                group.color: ranks[frozenset(group.words)] for group in puzzle.groups
            }
        choice = ranked[0]
        result = env.step(choice.words)
        observation = result.observation
        total_reward += result.reward
        feedback = observation.guess_history[-1]
        guesses.append({
            "words": list(choice.words),
            "clue_score": round(choice.value, 6),
            "feedback": feedback.feedback,
            "reward": result.reward,
            "revealed_color": feedback.revealed_group.color if feedback.revealed_group else None,
        })
    return {
        "puzzle_id": puzzle.puzzle_id,
        "status": observation.status,
        "groups_solved": len(observation.solved_groups),
        "mistakes": 4 - observation.mistakes_left,
        "turns": len(guesses),
        "total_reward": round(total_reward, 6),
        "initial_true_group_ranks": initial_true_group_ranks,
        "guesses": guesses,
    }


def evaluate(paths: list[Path], baseline: ClueBaseline, seed: int) -> dict:
    results = []
    for index, path in enumerate(paths, 1):
        results.append(evaluate_one(path, baseline, seed))
        if index % 10 == 0 or index == len(paths):
            print(f"Evaluated {index}/{len(paths)} puzzles", flush=True)
    color_counts = Counter(
        guess["revealed_color"] for result in results for guess in result["guesses"]
        if guess["revealed_color"] is not None
    )
    wins = sum(result["status"] == "won" for result in results)
    true_ranks = [rank for result in results
                  for rank in result["initial_true_group_ranks"].values()]
    return {
        "count": len(results),
        "wins": wins,
        "win_rate": wins / len(results) if results else 0.0,
        "mean_groups_solved": sum(result["groups_solved"] for result in results) / len(results)
        if results else 0.0,
        "mean_reward": sum(result["total_reward"] for result in results) / len(results)
        if results else 0.0,
        "solved_groups_by_color": dict(sorted(color_counts.items())),
        "initial_true_groups_top_10": sum(rank <= 10 for rank in true_ranks),
        "initial_true_groups_top_50": sum(rank <= 50 for rank in true_ranks),
        "initial_true_group_median_rank": median(true_ranks) if true_ranks else None,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puzzles", type=Path, default=Path("data/puzzles/nyt"))
    parser.add_argument("--output", type=Path, default=Path("reports/wordnet_baseline.json"))
    parser.add_argument("--fasttext", type=Path)
    parser.add_argument("--vector-cache", type=Path)
    parser.add_argument("--wordnet-data", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, help="evaluate only the first N sorted puzzles")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.fasttext and args.vector_cache:
        parser.error("choose either --fasttext or --vector-cache")
    paths = sorted(path for path in args.puzzles.glob("*.json") if path.name != "manifest.json")
    if args.limit is not None:
        paths = paths[:args.limit]
    if not paths:
        parser.error(f"no puzzle files found in {args.puzzles}")

    try:
        import nltk
    except ImportError:
        parser.error("NLTK is required; install it with pip install -e '.[wordnet]'")
    local_data = Path(__file__).resolve().parent.parent / ".venv/nltk_data"
    data_dir = args.wordnet_data or (local_data if local_data.exists() else None)
    if data_dir is not None:
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
    report = evaluate(paths, baseline, args.seed)
    manifest = args.puzzles / "manifest.json"
    report.update({
        "baseline": "wordnet+forms+fixed_vectors"
        if args.fasttext or args.vector_cache else "wordnet+forms",
        "vector_source": "fasttext_binary" if args.fasttext
        else "fasttext_subset" if args.vector_cache else None,
        "fasttext_source_sha256": vectors.metadata.get("source_sha256")
        if isinstance(vectors, FastTextSubsetVectors) else None,
        "seed": args.seed,
        "puzzles_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
        if manifest.exists() else None,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wins: {report['wins']}/{report['count']} ({report['win_rate']:.1%}); "
        f"mean groups: {report['mean_groups_solved']:.2f}; "
        f"mean reward: {report['mean_reward']:+.2f}"
    )
    print(f"Saved report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
