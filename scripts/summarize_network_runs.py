"""Compare warm-start and DQN validation results across training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def summarize(paths: list[Path]) -> dict:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len({run["seed"] for run in runs}) != len(runs):
        raise ValueError("training seeds must be unique")
    if len({run["manifest_sha256"] for run in runs}) != 1:
        raise ValueError("runs use different puzzle manifests")
    rows = []
    expected_ids = None
    for path, run in zip(paths, runs):
        before = run["validation_after_supervised"]
        after = run["validation_after_rl"]
        if after is None:
            raise ValueError(f"run has no DQN validation: {path}")
        ids = [item["puzzle_id"] for item in before["results"]]
        if ids != [item["puzzle_id"] for item in after["results"]]:
            raise ValueError(f"stages use different puzzles: {path}")
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError("runs use different validation puzzles")
        rows.append({
            "seed": run["seed"], "report": str(path),
            "warm_wins": before["wins"], "dqn_wins": after["wins"],
            "warm_mean_groups": before["mean_groups"],
            "dqn_mean_groups": after["mean_groups"],
            "warm_mean_reward": before["mean_reward"],
            "dqn_mean_reward": after["mean_reward"],
            "win_change": after["wins"] - before["wins"],
            "mean_group_change": after["mean_groups"] - before["mean_groups"],
            "mean_reward_change": after["mean_reward"] - before["mean_reward"],
            "warm_zero_group_puzzles": before["diagnostics"]["zero_group_puzzles"],
            "dqn_zero_group_puzzles": after["diagnostics"]["zero_group_puzzles"],
            "warm_purple_median_rank": before["diagnostics"]["initial_rank_by_color"]["purple"]["median_rank"],
            "dqn_purple_median_rank": after["diagnostics"]["initial_rank_by_color"]["purple"]["median_rank"],
            "warm_initial_true_groups_top_10": before["diagnostics"]["initial_true_groups_top_10"],
            "dqn_initial_true_groups_top_10": after["diagnostics"]["initial_true_groups_top_10"],
            "warm_incorrect_followup_overlap_at_least_3": before["diagnostics"]["wrong_guess_followups"]["incorrect"]["next_guess_overlap_at_least_3"],
            "dqn_incorrect_followup_overlap_at_least_3": after["diagnostics"]["wrong_guess_followups"]["incorrect"]["next_guess_overlap_at_least_3"],
            "warm_incorrect_followup_count": before["diagnostics"]["wrong_guess_followups"]["incorrect"]["count"],
            "dqn_incorrect_followup_count": after["diagnostics"]["wrong_guess_followups"]["incorrect"]["count"],
            "warm_one_away_next_correct": before["diagnostics"]["wrong_guess_followups"]["one_away"]["next_guess_correct"],
            "dqn_one_away_next_correct": after["diagnostics"]["wrong_guess_followups"]["one_away"]["next_guess_correct"],
            "warm_one_away_followup_count": before["diagnostics"]["wrong_guess_followups"]["one_away"]["count"],
            "dqn_one_away_followup_count": after["diagnostics"]["wrong_guess_followups"]["one_away"]["count"],
        })
    keys = ("warm_wins", "dqn_wins", "win_change", "warm_mean_groups",
            "dqn_mean_groups", "mean_group_change", "warm_mean_reward",
            "dqn_mean_reward", "mean_reward_change")
    consistency = {}
    for stage in ("validation_after_supervised", "validation_after_rl"):
        result_lists = [run[stage]["results"] for run in runs]
        win_counts = [sum(row["status"] == "won" for row in same_puzzle)
                      for same_puzzle in zip(*result_lists)]
        consistency[stage] = {
            "won_by_no_seed": sum(count == 0 for count in win_counts),
            "won_by_any_seed": sum(count > 0 for count in win_counts),
            "won_by_every_seed": sum(count == len(runs) for count in win_counts),
        }
    return {
        "validation_count": len(expected_ids or []),
        "test_count_used": 0,
        "manifest_sha256": runs[0]["manifest_sha256"],
        "seeds": rows,
        "win_consistency": consistency,
        "across_seeds": {
            key: {"mean": mean(row[key] for row in rows),
                  "min": min(row[key] for row in rows),
                  "max": max(row[key] for row in rows)}
            for key in keys
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("reports/network_diagnostics_summary.json"))
    args = parser.parse_args()
    result = summarize(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Compared {len(result['seeds'])} seeds on "
          f"{result['validation_count']} validation puzzles; saved {args.output}")


if __name__ == "__main__":
    main()
