"""Play one word puzzle with a trained candidate Q-network."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .environment import ConnectionsEnv
from .knowledge import FastTextSubsetVectors, WordNetKnowledgeCache
from .meanings import MeaningClueProvider
from .network import CandidateQNetwork, load_or_build_features, score_candidates
from .puzzle import load_puzzle
from .word_forms import WordFormClueProvider


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puzzle", type=Path,
                        default=Path("data/puzzles/nyt/nyt-2024-10-01.json"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(".venv/checkpoints/candidate_q.pt"))
    parser.add_argument("--vectors", type=Path,
                        default=Path(".venv/models/cc.en.300.all-718.sqlite3"))
    parser.add_argument("--feature-cache", type=Path,
                        default=Path(".venv/features/nyt-v1"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    import nltk
    local_corpus = Path(__file__).resolve().parent.parent / ".venv/nltk_data"
    if local_corpus.exists():
        nltk.data.path.insert(0, str(local_corpus))
    vectors = FastTextSubsetVectors(args.vectors)
    knowledge = WordNetKnowledgeCache.from_nltk(vectors)
    features = load_or_build_features(
        args.puzzle, args.feature_cache, knowledge,
        MeaningClueProvider(knowledge), WordFormClueProvider(knowledge.contains),
    )
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved["vector_source_sha256"] != vectors.metadata.get("source_sha256"):
        raise ValueError("checkpoint and fixed vector cache use different sources")
    model = CandidateQNetwork().eval()
    model.load_state_dict(saved["model_state"])
    env = ConnectionsEnv(load_puzzle(args.puzzle))
    observation = env.reset(seed=args.seed)
    total_reward = 0.0
    while observation.status == "in_progress":
        legal = features.legal_indices(observation)
        with torch.no_grad():
            scores = score_candidates(model, features, observation, legal, vectors)
        index = int(legal[int(torch.argmax(scores))])
        guess = features.guess(index)
        result = env.step(guess)
        total_reward += result.reward
        last = result.observation.guess_history[-1]
        print(f"{', '.join(guess)} -> {last.feedback}; reward {result.reward:+g}")
        if last.revealed_group is not None:
            print(f"  Revealed: {last.revealed_group.title} ({last.revealed_group.color})")
        observation = result.observation
    print(f"{observation.status}: {len(observation.solved_groups)} groups, "
          f"{len(observation.guess_history)} turns, reward {total_reward:+g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
