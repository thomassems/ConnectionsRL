"""Compare LLM solver reports: variance across repeats and across models.

Groups the given reports by (provider, model, effort). Repeated reports of
the same configuration become a variance estimate; distinct configurations
become a leaderboard. All reports must cover the same puzzle set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

COLORS = ("yellow", "green", "blue", "purple")


def _puzzle_ids(report: dict) -> list[str]:
    return sorted(result["puzzle_id"] for result in report["results"])


def _config(report: dict) -> tuple[str, str, str]:
    return (
        report.get("provider", "anthropic"),
        report["model"],
        str(report.get("effort")),
    )


def _color_rate(report: dict, color: str) -> float:
    by_color = report.get("solved_rate_by_color")
    if by_color and color in by_color:
        return by_color[color]["rate"]
    # Fall back to the older count-only field.
    counts = report.get("solved_groups_by_color", {})
    return counts.get(color, 0) / report["count"] if report["count"] else 0.0


def _spread(values: list[float]) -> dict:
    return {
        "mean": mean(values),
        "min": min(values),
        "max": max(values),
        "stdev": pstdev(values) if len(values) > 1 else 0.0,
    }


def summarize(paths: list[Path]) -> dict:
    reports = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    puzzle_ids = _puzzle_ids(reports[0][1])
    for path, report in reports:
        if _puzzle_ids(report) != puzzle_ids:
            raise ValueError(f"reports cover different puzzles: {path}")
    manifests = {r.get("manifest_sha256") for _, r in reports if r.get("manifest_sha256")}
    if len(manifests) > 1:
        raise ValueError("reports use different puzzle manifests")

    configs: dict[tuple, list[tuple[Path, dict]]] = {}
    for path, report in reports:
        configs.setdefault(_config(report), []).append((path, report))

    rows = []
    for (provider, model, effort), runs in configs.items():
        run_reports = [report for _, report in runs]
        # Across repeats: how deterministic is a win?
        statuses = {pid: [] for pid in puzzle_ids}
        for report in run_reports:
            for result in report["results"]:
                statuses[result["puzzle_id"]].append(result["status"] == "won")
        won_flags = list(statuses.values())
        rows.append({
            "provider": provider,
            "model": model,
            "effort": None if effort == "None" else effort,
            "runs": len(runs),
            "reports": [str(path) for path, _ in runs],
            "wins": _spread([r["wins"] for r in run_reports]),
            "win_rate": _spread([r["win_rate"] for r in run_reports]),
            "mean_groups_solved": _spread([r["mean_groups_solved"] for r in run_reports]),
            "mean_reward": _spread([r["mean_reward"] for r in run_reports]),
            "color_solve_rate": {
                color: _spread([_color_rate(r, color) for r in run_reports])
                for color in COLORS
            },
            "fallback_guesses": _spread([r.get("fallback_guesses", 0) for r in run_reports]),
            "total_estimated_cost_usd": round(sum(
                r.get("estimated_cost_usd") or 0.0 for r in run_reports
            ), 4),
            "win_consistency": {
                "won_by_every_run": sum(all(flags) for flags in won_flags),
                "won_by_some_run": sum(any(flags) for flags in won_flags),
                "won_by_no_run": sum(not any(flags) for flags in won_flags),
            },
        })
    rows.sort(key=lambda row: row["wins"]["mean"], reverse=True)
    return {
        "puzzle_count": len(puzzle_ids),
        "test_count_used": 0,
        "manifest_sha256": next(iter(manifests), None),
        "configurations": rows,
    }


def _print_table(result: dict) -> None:
    header = (
        f"{'provider/model':28} {'eff':4} {'runs':>4} {'wins(mean)':>10} "
        f"{'win%':>6} {'groups':>7} {'reward':>7} {'purple%':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in result["configurations"]:
        name = f"{row['provider']}/{row['model']}"[:28]
        print(
            f"{name:28} {str(row['effort'] or '-'):4} {row['runs']:>4} "
            f"{row['wins']['mean']:>10.1f} {row['win_rate']['mean']*100:>5.1f}% "
            f"{row['mean_groups_solved']['mean']:>7.2f} {row['mean_reward']['mean']:>+7.2f} "
            f"{row['color_solve_rate']['purple']['mean']*100:>7.1f}%"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/llm_runs_summary.json"))
    args = parser.parse_args(argv)
    result = summarize(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _print_table(result)
    print(f"\nCompared {len(result['configurations'])} configuration(s) on "
          f"{result['puzzle_count']} puzzles; saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
