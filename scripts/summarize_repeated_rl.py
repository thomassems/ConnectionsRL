"""Compare validation after each RL pass from repeated-training reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def _metrics(stage: dict) -> dict:
    diagnostics = stage["diagnostics"]
    purple = diagnostics["initial_rank_by_color"]["purple"]
    incorrect = diagnostics["wrong_guess_followups"]["incorrect"]
    one_away = diagnostics["wrong_guess_followups"]["one_away"]
    return {
        "wins": stage["wins"],
        "mean_groups": stage["mean_groups"],
        "mean_reward": stage["mean_reward"],
        "zero_group_puzzles": diagnostics["zero_group_puzzles"],
        "purple_top_10": purple["top_10"],
        "purple_median_rank": purple["median_rank"],
        "incorrect_followup_overlap_at_least_3": incorrect["next_guess_overlap_at_least_3"],
        "incorrect_followup_count": incorrect["count"],
        "one_away_next_correct": one_away["next_guess_correct"],
        "one_away_followup_count": one_away["count"],
    }


def summarize(paths: list[Path]) -> dict:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len({run["seed"] for run in runs}) != len(runs):
        raise ValueError("training seeds must be unique")
    if len({run["manifest_sha256"] for run in runs}) != 1:
        raise ValueError("runs use different puzzle manifests")
    if any(run["test_count_used"] != 0 for run in runs):
        raise ValueError("test puzzles must not be used in this comparison")
    expected_ids = None
    rows = []
    for path, run in zip(paths, runs):
        stages = [(0, run["validation_after_supervised"])] + [
            (entry["epoch"], entry) for entry in run["validation_by_rl_epoch"]
        ]
        if [epoch for epoch, _ in stages] != list(range(len(stages))):
            raise ValueError(f"missing or repeated RL epoch in {path}")
        if len(stages) < 3:
            raise ValueError(f"need at least two RL epochs in {path}")
        for epoch, stage in stages:
            ids = [result["puzzle_id"] for result in stage["results"]]
            if expected_ids is None:
                expected_ids = ids
            elif ids != expected_ids:
                raise ValueError(f"validation puzzle IDs differ in {path}, epoch {epoch}")
        rows.append({
            "seed": run["seed"], "report": str(path),
            "stages": [
                {"rl_epochs": epoch, **_metrics(stage)} for epoch, stage in stages
            ],
        })
    epoch_count = len(rows[0]["stages"])
    if any(len(row["stages"]) != epoch_count for row in rows):
        raise ValueError("runs have different numbers of RL epochs")
    across_seeds = []
    for epoch in range(epoch_count):
        metrics = rows[0]["stages"][epoch]
        across_seeds.append({
            "rl_epochs": epoch,
            **{
                key: {"mean": mean(row["stages"][epoch][key] for row in rows),
                      "min": min(row["stages"][epoch][key] for row in rows),
                      "max": max(row["stages"][epoch][key] for row in rows)}
                for key in ("wins", "mean_groups", "mean_reward", "zero_group_puzzles",
                            "purple_top_10", "purple_median_rank")
            },
        })
    return {
        "validation_count": len(expected_ids or []),
        "test_count_used": 0,
        "manifest_sha256": runs[0]["manifest_sha256"],
        "seeds": rows,
        "across_seeds": across_seeds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("reports/repeated_rl_summary.json"))
    args = parser.parse_args()
    result = summarize(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Compared {len(result['seeds'])} seeds after "
          f"{len(result['across_seeds']) - 1} RL passes on "
          f"{result['validation_count']} validation puzzles")


if __name__ == "__main__":
    main()
