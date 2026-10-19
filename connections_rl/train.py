"""Train the candidate scorer with labeled groups, then game rewards."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import median

import numpy as np
import torch
from torch.nn import functional as F

from .environment import ConnectionsEnv, Observation
from .knowledge import FastTextSubsetVectors, WordNetKnowledgeCache
from .meanings import MeaningClueProvider
from .network import CandidateQNetwork, PuzzleFeatures, load_or_build_features, score_candidates
from .puzzle import load_puzzle
from .word_forms import WordFormClueProvider


@dataclass(frozen=True)
class Transition:
    puzzle_id: str
    before: Observation
    action: int
    reward: float
    after: Observation


def _split_paths(root: Path, split_path: Path) -> tuple[list[Path], list[Path], str]:
    data = json.loads(split_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    if data["manifest_sha256"] != digest:
        raise ValueError("split manifest hash does not match puzzle collection")
    return (
        [root / f"{name}.json" for name in data["splits"]["train"]],
        [root / f"{name}.json" for name in data["splits"]["validation"]],
        digest,
    )


def _true_indices(path: Path, features: PuzzleFeatures) -> np.ndarray:
    puzzle = load_puzzle(path)
    by_words = {
        frozenset(features.words[int(word)] for word in row): index
        for index, row in enumerate(features.candidates)
    }
    return np.asarray([by_words[frozenset(group.words)] for group in puzzle.groups], dtype=np.int64)


def supervised_epoch(
    model: CandidateQNetwork, paths: list[Path], get_features,
    vectors: FastTextSubsetVectors, optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
) -> float:
    model.train()
    order = rng.permutation(len(paths))
    losses = []
    for position, path_index in enumerate(order, 1):
        path = paths[int(path_index)]
        features = get_features(path.stem)
        positives = _true_indices(path, features)
        clues = features.clues
        heuristic = clues[:, 2] + clues[:, 11] + clues[:, 16] + clues[:, -4:].max(axis=1)
        all_negatives = np.setdiff1d(np.arange(len(clues)), positives)
        hard = all_negatives[np.argsort(heuristic[all_negatives])[-64:]]
        random_negatives = rng.choice(all_negatives, size=64, replace=False)
        selected = np.unique(np.concatenate((positives, hard, random_negatives)))
        observation = ConnectionsEnv(load_puzzle(path)).reset(seed=7)
        scores = score_candidates(model, features, observation, selected, vectors)
        labels = torch.from_numpy(np.isin(selected, positives).astype(np.float32))
        weights = torch.where(labels > 0, 16.0, 1.0)
        loss = F.binary_cross_entropy_with_logits(scores, labels, weight=weights)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if position % 50 == 0 or position == len(paths):
            print(f"Supervised {position}/{len(paths)}; mean loss {np.mean(losses):.4f}", flush=True)
    return float(np.mean(losses))


def play_episode(
    path: Path, features: PuzzleFeatures, model: CandidateQNetwork,
    vectors: FastTextSubsetVectors, rng: random.Random, epsilon: float = 0.0,
    diagnostics: bool = False,
) -> tuple[dict, list[Transition]]:
    puzzle = load_puzzle(path)
    env = ConnectionsEnv(puzzle)
    observation = env.reset(seed=7)
    transitions = []
    trace = []
    total_reward = 0.0
    if diagnostics:
        candidate_by_words = {
            frozenset(features.guess(index)): index
            for index in range(len(features.candidates))
        }
    while observation.status == "in_progress":
        legal = features.legal_indices(observation)
        if not len(legal):
            raise RuntimeError(f"no legal candidate for {path.stem}")
        with torch.no_grad():
            scores = score_candidates(model, features, observation, legal, vectors)
        if rng.random() < epsilon:
            top = torch.topk(scores, min(50, len(legal))).indices.numpy()
            action = int(legal[int(rng.choice(top))])
        else:
            action = int(legal[int(torch.argmax(scores))])
        if diagnostics:
            order = np.argsort(-scores.numpy(), kind="stable")
            ranks = np.empty(len(legal), dtype=np.int32)
            ranks[order] = np.arange(1, len(legal) + 1)
            rank_by_index = dict(zip(legal.tolist(), ranks.tolist()))
            true_groups = {
                group.color: {
                    "title": group.title,
                    "rank": rank_by_index[candidate_by_words[frozenset(group.words)]],
                    "q_value": round(float(scores[np.where(legal == candidate_by_words[frozenset(group.words)])[0][0]]), 5),
                }
                for group in puzzle.groups if group not in observation.solved_groups
            }
            row = {
                "turn": len(transitions) + 1,
                "mistakes_left_before": observation.mistakes_left,
                "legal_candidates": len(legal),
                "guess": list(features.guess(action)),
                "chosen_rank": rank_by_index[action],
                "chosen_q_value": round(float(scores[np.where(legal == action)[0][0]]), 5),
                "chosen_meaning_strength": round(float(features.clues[action, 2]), 5),
                "chosen_word_form_strength": round(float(features.clues[action, 18]), 5),
                "true_group_ranks": true_groups,
            }
        result = env.step(features.guess(action))
        transitions.append(Transition(path.stem, observation, action, result.reward, result.observation))
        if diagnostics:
            feedback = result.observation.guess_history[-1]
            row.update({
                "feedback": feedback.feedback,
                "reward": result.reward,
                "revealed_color": feedback.revealed_group.color if feedback.revealed_group else None,
                "max_overlap_with_unsolved_group": max(
                    len(set(features.guess(action)).intersection(group.words))
                    for group in puzzle.groups if group not in observation.solved_groups
                ),
            })
            trace.append(row)
        total_reward += result.reward
        observation = result.observation
    result = {
        "puzzle_id": path.stem, "status": observation.status,
        "groups_solved": len(observation.solved_groups),
        "reward": round(total_reward, 6), "turns": len(transitions),
    }
    if diagnostics:
        result["diagnostic_turns"] = trace
    return result, transitions


def summarize_diagnostics(results: list[dict]) -> dict:
    from collections import Counter

    initial = [result["diagnostic_turns"][0] for result in results]
    initial_ranks = [entry["rank"] for turn in initial
                     for entry in turn["true_group_ranks"].values()]
    ranks_by_color = {
        color: [turn["true_group_ranks"][color]["rank"] for turn in initial]
        for color in ("yellow", "green", "blue", "purple")
    }
    followups = []
    for result in results:
        turns = result["diagnostic_turns"]
        for before, after in zip(turns, turns[1:]):
            if before["feedback"] == "correct":
                continue
            prior_best = min(group["rank"] for group in before["true_group_ranks"].values())
            next_best = min(group["rank"] for group in after["true_group_ranks"].values())
            followups.append({
                "feedback": before["feedback"],
                "next_guess_overlap": len(set(before["guess"]).intersection(after["guess"])),
                "best_true_rank_improvement": prior_best - next_best,
                "next_guess_correct": after["feedback"] == "correct",
            })
    return {
        "initial_true_groups_top_10": sum(rank <= 10 for rank in initial_ranks),
        "initial_true_groups_top_50": sum(rank <= 50 for rank in initial_ranks),
        "initial_true_group_median_rank": median(initial_ranks),
        "initial_rank_by_color": {
            color: {"top_10": sum(rank <= 10 for rank in ranks),
                    "median_rank": median(ranks)}
            for color, ranks in ranks_by_color.items()
        },
        "first_guess_feedback": dict(Counter(turn["feedback"] for turn in initial)),
        "zero_group_puzzles": sum(result["groups_solved"] == 0 for result in results),
        "zero_group_with_initial_true_top_10": sum(
            result["groups_solved"] == 0
            and min(group["rank"] for group in result["diagnostic_turns"][0]["true_group_ranks"].values()) <= 10
            for result in results
        ),
        "wrong_guess_followups": {
            feedback: {
                "count": len(items),
                "next_guess_correct": sum(item["next_guess_correct"] for item in items),
                "next_guess_overlap_at_least_3": sum(item["next_guess_overlap"] >= 3 for item in items),
                "mean_best_true_rank_improvement": float(np.mean([
                    item["best_true_rank_improvement"] for item in items
                ])) if items else None,
            }
            for feedback in ("one_away", "incorrect")
            for items in ([item for item in followups if item["feedback"] == feedback],)
        },
    }


def evaluate(
    paths: list[Path], get_features, model: CandidateQNetwork,
    vectors: FastTextSubsetVectors,
) -> dict:
    model.eval()
    results = []
    for position, path in enumerate(paths, 1):
        result, _ = play_episode(path, get_features(path.stem), model, vectors,
                                 random.Random(0), diagnostics=True)
        results.append(result)
        if position % 25 == 0 or position == len(paths):
            print(f"Validation {position}/{len(paths)}", flush=True)
    return {
        "count": len(results),
        "wins": sum(item["status"] == "won" for item in results),
        "mean_groups": float(np.mean([item["groups_solved"] for item in results])),
        "mean_reward": float(np.mean([item["reward"] for item in results])),
        "diagnostics": summarize_diagnostics(results),
        "results": results,
    }


def reinforcement_epoch(
    model: CandidateQNetwork, paths: list[Path], get_features,
    vectors: FastTextSubsetVectors, optimizer: torch.optim.Optimizer,
    seed: int, epsilon: float = 0.15, gamma: float = 0.99,
) -> dict:
    rng = random.Random(seed)
    target = copy.deepcopy(model).eval()
    replay: deque[Transition] = deque(maxlen=256)
    episode_results = []
    losses = []
    order = paths.copy()
    rng.shuffle(order)
    for position, path in enumerate(order, 1):
        model.eval()
        result, transitions = play_episode(path, get_features(path.stem), model, vectors, rng, epsilon)
        episode_results.append(result)
        replay.extend(transitions)
        if len(replay) >= 16:
            sample = rng.sample(list(replay), 16)
            predictions = []
            targets = []
            model.train()
            for item in sample:
                features = get_features(item.puzzle_id)
                predictions.append(score_candidates(
                    model, features, item.before, np.asarray([item.action]), vectors,
                )[0])
                with torch.no_grad():
                    legal = features.legal_indices(item.after)
                    future = float(score_candidates(target, features, item.after, legal, vectors).max()) if len(legal) else 0.0
                targets.append(item.reward + gamma * future)
            loss = F.smooth_l1_loss(torch.stack(predictions), torch.tensor(targets))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if position % 25 == 0:
            target.load_state_dict(model.state_dict())
        if position % 50 == 0 or position == len(paths):
            print(f"RL {position}/{len(paths)}; recent mean groups "
                  f"{np.mean([item['groups_solved'] for item in episode_results[-50:]]):.2f}", flush=True)
    return {
        "episodes": len(episode_results),
        "exploration_epsilon": epsilon,
        "mean_groups": float(np.mean([item["groups_solved"] for item in episode_results])),
        "wins": sum(item["status"] == "won" for item in episode_results),
        "mean_update_loss": float(np.mean(losses)) if losses else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puzzles", type=Path, default=Path("data/puzzles/nyt"))
    parser.add_argument("--split", type=Path, default=Path("data/splits/nyt-v1.json"))
    parser.add_argument("--vectors", type=Path, default=Path(".venv/models/cc.en.300.all-718.sqlite3"))
    parser.add_argument("--feature-cache", type=Path, default=Path(".venv/features/nyt-v1"))
    parser.add_argument("--output", type=Path, default=Path(".venv/checkpoints/candidate_q.pt"))
    parser.add_argument("--report", type=Path, default=Path("reports/network_training.json"))
    parser.add_argument("--resume-supervised", type=Path,
                        help="resume DQN from a saved supervised checkpoint")
    parser.add_argument("--evaluate-checkpoint", type=Path,
                        help="only rerun validation diagnostics for a saved checkpoint")
    parser.add_argument("--supervised-epochs", type=int, default=3)
    parser.add_argument("--rl-epochs", type=int, default=1)
    parser.add_argument("--evaluate-each-rl-epoch", action="store_true",
                        help="save validation results and a checkpoint after each RL pass")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if args.supervised_epochs < 0 or args.rl_epochs < 0:
        parser.error("epoch counts cannot be negative")
    if not args.supervised_epochs and not args.resume_supervised and not args.evaluate_checkpoint:
        parser.error("this first run requires a supervised warm start")
    import nltk
    local_corpus = Path(__file__).resolve().parent.parent / ".venv/nltk_data"
    if local_corpus.exists():
        nltk.data.path.insert(0, str(local_corpus))
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    paths, validation_paths, manifest_hash = _split_paths(args.puzzles, args.split)
    if args.train_limit:
        paths = paths[:args.train_limit]
    if args.validation_limit:
        validation_paths = validation_paths[:args.validation_limit]
    vectors = FastTextSubsetVectors(args.vectors)
    knowledge = WordNetKnowledgeCache.from_nltk(vectors)
    meanings = MeaningClueProvider(knowledge)
    forms = WordFormClueProvider(knowledge.contains)

    @lru_cache(maxsize=64)
    def get_features(puzzle_id: str) -> PuzzleFeatures:
        return load_or_build_features(
            args.puzzles / f"{puzzle_id}.json", args.feature_cache,
            knowledge, meanings, forms,
        )

    model = CandidateQNetwork()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    training = []
    if args.evaluate_checkpoint:
        saved = torch.load(args.evaluate_checkpoint, map_location="cpu", weights_only=True)
        if saved["manifest_sha256"] != manifest_hash:
            raise ValueError("checkpoint manifest does not match current split")
        if saved["vector_source_sha256"] != vectors.metadata.get("source_sha256"):
            raise ValueError("checkpoint vectors do not match current vector cache")
        model.load_state_dict(saved["model_state"])
        evaluation = evaluate(validation_paths, get_features, model, vectors)
        report = {
            "method": "greedy validation diagnostics for saved candidate scorer",
            "checkpoint": str(args.evaluate_checkpoint),
            "seed": saved["seed"], "validation_count": len(validation_paths),
            "test_count_used": 0, "manifest_sha256": manifest_hash,
            "validation": evaluation,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved diagnostic report {args.report}")
        return 0
    if args.resume_supervised:
        saved = torch.load(args.resume_supervised, map_location="cpu", weights_only=True)
        if saved["manifest_sha256"] != manifest_hash:
            raise ValueError("checkpoint manifest does not match current split")
        if saved["vector_source_sha256"] != vectors.metadata.get("source_sha256"):
            raise ValueError("checkpoint vectors do not match current vector cache")
        model.load_state_dict(saved["model_state"])
    else:
        for epoch in range(args.supervised_epochs):
            loss = supervised_epoch(model, paths, get_features, vectors, optimizer, rng)
            training.append({"phase": "supervised", "epoch": epoch + 1, "loss": loss})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        warm_path = args.output.with_name(args.output.stem + ".supervised" + args.output.suffix)
        torch.save({
            "model_state": model.state_dict(), "seed": args.seed,
            "manifest_sha256": manifest_hash,
            "vector_source_sha256": vectors.metadata.get("source_sha256"),
        }, warm_path)
    before_rl = evaluate(validation_paths, get_features, model, vectors)
    validation_by_rl_epoch = []
    for epoch in range(args.rl_epochs):
        rl_result = reinforcement_epoch(
            model, paths, get_features, vectors, optimizer, args.seed + epoch,
        )
        training.append({"phase": "reinforcement", "epoch": epoch + 1, **rl_result})
        if args.evaluate_each_rl_epoch:
            evaluation = evaluate(validation_paths, get_features, model, vectors)
            validation_by_rl_epoch.append({"epoch": epoch + 1, **evaluation})
            epoch_path = args.output.with_name(
                args.output.stem + f".epoch-{epoch + 1}" + args.output.suffix
            )
            epoch_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(), "seed": args.seed,
                "manifest_sha256": manifest_hash,
                "vector_source_sha256": vectors.metadata.get("source_sha256"),
            }, epoch_path)
    if args.rl_epochs:
        after_rl = (validation_by_rl_epoch[-1] if args.evaluate_each_rl_epoch
                    else evaluate(validation_paths, get_features, model, vectors))
    else:
        after_rl = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(), "seed": args.seed,
        "manifest_sha256": manifest_hash,
        "vector_source_sha256": vectors.metadata.get("source_sha256"),
    }, args.output)
    report = {
        "method": "candidate scorer: supervised warm start, then DQN updates",
        "seed": args.seed, "train_count": len(paths),
        "validation_count": len(validation_paths),
        "test_count_used": 0,
        "manifest_sha256": manifest_hash,
        "vector_source_sha256": vectors.metadata.get("source_sha256"),
        "training": training,
        "validation_after_supervised": before_rl,
        "validation_after_rl": after_rl,
    }
    if args.evaluate_each_rl_epoch:
        report["validation_by_rl_epoch"] = validation_by_rl_epoch
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved checkpoint {args.output} and report {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
