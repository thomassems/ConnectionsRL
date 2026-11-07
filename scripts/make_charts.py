"""Draw the README result charts as SVG from the saved reports in ``reports/``.

Every number in the charts comes from a committed report, so the figures can be
regenerated after a rerun and nothing in the README is typed by hand::

    python3 scripts/make_charts.py

Output goes to ``docs/assets/``. The script uses only the standard library.

The SVGs carry a light palette by default and swap to a dark one under
``prefers-color-scheme: dark``. Each one paints its own background, so a viewer
that ignores the media query still shows dark text on a light card rather than
dark text on a dark page.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS = REPO_ROOT / "reports"
ASSETS = REPO_ROOT / "docs" / "assets"

VALIDATION_COUNT = 141
COLORS = ("yellow", "green", "blue", "purple")

# Agent series colors, held constant across every chart so a reader learns them
# once. Chosen to stay distinguishable on both backgrounds.
SERIES = {
    "baseline": "#8c8f94",
    "vectors": "#5b8def",
    "network": "#e0922f",
    "llm": "#1f9d6b",
}
PUZZLE_COLORS = {
    "yellow": "#eab308",
    "green": "#22a06b",
    "blue": "#3b82f6",
    "purple": "#a855f7",
}

STYLE = """
  .bg { fill: #ffffff; }
  .title { fill: #1f2328; font: 600 17px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
  .sub { fill: #656d76; font: 400 12.5px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
  .label { fill: #1f2328; font: 500 13px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
  .value { fill: #1f2328; font: 600 13px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
  .tick { fill: #656d76; font: 400 11.5px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
  .grid { stroke: #d8dee4; stroke-width: 1; }
  .axis { stroke: #8c959f; stroke-width: 1; }
  .whisker { stroke: #1f2328; stroke-width: 1.5; }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #0d1117; }
    .title { fill: #e6edf3; }
    .sub { fill: #9198a1; }
    .label { fill: #e6edf3; }
    .value { fill: #e6edf3; }
    .tick { fill: #9198a1; }
    .grid { stroke: #30363d; }
    .axis { stroke: #6e7681; }
    .whisker { stroke: #e6edf3; }
  }
"""


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img">\n'
        f"<style>{STYLE}</style>\n"
        f'<rect class="bg" x="0" y="0" width="{width}" height="{height}" rx="6"/>\n'
        f"{body}\n</svg>\n"
    )


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


# --- data assembly ---------------------------------------------------------


def solved_by_color_from_counts(report: dict) -> dict[str, float]:
    """Solve rate per color from a report's ``solved_groups_by_color`` counts."""
    counts = report.get("solved_groups_by_color") or {}
    total = report["count"]
    return {color: counts.get(color, 0) / total for color in COLORS}


def solved_by_color_from_turns(validation: dict) -> dict[str, float]:
    """Solve rate per color by counting revealed colors in the per-turn traces."""
    counts = dict.fromkeys(COLORS, 0)
    for result in validation["results"]:
        for turn in result["diagnostic_turns"]:
            revealed = turn.get("revealed_color")
            if revealed in counts:
                counts[revealed] += 1
    total = validation["count"]
    return {color: counts[color] / total for color in COLORS}


def solved_by_color_from_rates(report: dict) -> dict[str, float]:
    rates = report["solved_rate_by_color"]
    return {color: rates[color]["rate"] for color in COLORS}


def gather() -> dict:
    wordnet = load("wordnet_baseline_validation.json")
    vectors = load("fasttext_wordnet_baseline_validation.json")
    network_reports = [
        load("network_training_seed7_diagnostics.json"),
        load("network_training_seed17.json"),
        load("network_training_seed27.json"),
    ]
    llm_reports = [
        load(f"deepseek_chat_val_{index}.json") for index in (1, 2, 3)
    ]
    repeated = load("repeated_rl_summary.json")

    for report in (wordnet, vectors, *llm_reports):
        if report["count"] != VALIDATION_COUNT:
            raise SystemExit(f"expected {VALIDATION_COUNT} puzzles, got {report['count']}")

    network_wins = [report["validation_after_rl"]["wins"] for report in network_reports]
    llm_wins = [report["wins"] for report in llm_reports]

    network_color = [
        solved_by_color_from_turns(report["validation_after_rl"])
        for report in network_reports
    ]
    llm_color = [solved_by_color_from_rates(report) for report in llm_reports]

    return {
        "wordnet": wordnet,
        "vectors": vectors,
        "network_wins": network_wins,
        "llm_wins": llm_wins,
        "by_color": {
            "baseline": solved_by_color_from_counts(wordnet),
            "vectors": solved_by_color_from_counts(vectors),
            "network": {
                color: mean(run[color] for run in network_color) for color in COLORS
            },
            "llm": {color: mean(run[color] for run in llm_color) for color in COLORS},
        },
        "repeated": repeated,
    }


# --- charts ----------------------------------------------------------------


def chart_wins(data: dict, path: Path) -> None:
    """Horizontal bars: puzzles won out of the 141 validation puzzles."""
    rows = [
        ("Rule baseline", "WordNet + spelling clues", data["wordnet"]["wins"], None, "baseline"),
        ("+ fastText vectors", "same rules, plus vector similarity", data["vectors"]["wins"], None, "vectors"),
        (
            "Trained network",
            "321k-param scorer, mean of 3 seeds",
            mean(data["network_wins"]),
            (min(data["network_wins"]), max(data["network_wins"])),
            "network",
        ),
        (
            "Language model",
            "deepseek-chat, mean of 3 runs",
            mean(data["llm_wins"]),
            (min(data["llm_wins"]), max(data["llm_wins"])),
            "llm",
        ),
    ]

    width, height = 760, 310
    left, top = 196, 78
    plot_width = 468
    row_height = 46
    bar_height = 22
    max_value = 60  # a little headroom above the best run

    def x_of(value: float) -> float:
        return left + plot_width * value / max_value

    parts = [
        f'<text class="title" x="24" y="34">Puzzles won, out of 141</text>',
        f'<text class="sub" x="24" y="56">Same 141 validation puzzles for every agent. '
        f"Held-out test puzzles were not used.</text>",
    ]

    for tick in range(0, max_value + 1, 10):
        x = x_of(tick)
        parts.append(
            f'<line class="grid" x1="{x:.1f}" y1="{top - 6}" '
            f'x2="{x:.1f}" y2="{top + row_height * len(rows) - 14}"/>'
        )
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{top + row_height * len(rows) + 4}" '
            f'text-anchor="middle">{tick}</text>'
        )

    for index, (name, note, value, spread, key) in enumerate(rows):
        y = top + index * row_height
        bar_y = y + (row_height - bar_height) / 2 - 10
        parts.append(f'<text class="label" x="24" y="{bar_y + 11:.1f}">{esc(name)}</text>')
        parts.append(f'<text class="sub" x="24" y="{bar_y + 27:.1f}">{esc(note)}</text>')
        bar_width = max(x_of(value) - left, 1.0)
        parts.append(
            f'<rect x="{left}" y="{bar_y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height}" rx="3" fill="{SERIES[key]}"/>'
        )
        if spread is not None:
            low, high = x_of(spread[0]), x_of(spread[1])
            mid = bar_y + bar_height / 2
            parts.append(
                f'<line class="whisker" x1="{low:.1f}" y1="{mid:.1f}" x2="{high:.1f}" y2="{mid:.1f}"/>'
            )
            for edge in (low, high):
                parts.append(
                    f'<line class="whisker" x1="{edge:.1f}" y1="{mid - 5:.1f}" '
                    f'x2="{edge:.1f}" y2="{mid + 5:.1f}"/>'
                )
            text = f"{value:.1f}  ({spread[0]}–{spread[1]})"
            text_x = high + 12
        else:
            text = f"{value:.0f}"
            text_x = x_of(value) + 12
        parts.append(
            f'<text class="value" x="{text_x:.1f}" y="{bar_y + 16:.1f}">{text}</text>'
        )

    baseline_y = top + row_height * len(rows) - 14
    parts.append(
        f'<line class="axis" x1="{left}" y1="{top - 6}" x2="{left}" y2="{baseline_y}"/>'
    )
    parts.append(
        '<text class="sub" x="24" y="296">Whiskers show the best and worst individual run; '
        "the rule baselines are deterministic.</text>"
    )
    path.write_text(svg(width, height, "\n".join(parts)), encoding="utf-8")


def chart_by_color(data: dict, path: Path) -> None:
    """Grouped bars: share of groups solved, split by the group's color."""
    series = [
        ("Rule baseline", "baseline"),
        ("+ vectors", "vectors"),
        ("Trained network", "network"),
        ("Language model", "llm"),
    ]
    by_color = data["by_color"]

    width, height = 760, 400
    left, top = 64, 96
    plot_width, plot_height = 660, 186
    group_width = plot_width / len(COLORS)
    bar_width = 26
    gap = 8
    bars_width = len(series) * bar_width + (len(series) - 1) * gap

    def y_of(rate: float) -> float:
        return top + plot_height * (1 - rate)

    parts = [
        '<text class="title" x="24" y="34">Share of groups solved, by group color</text>',
        '<text class="sub" x="24" y="56">Yellow is the easiest category and purple the '
        "trickiest, usually wordplay rather than meaning.</text>",
    ]

    for tick in (0, 25, 50, 75, 100):
        y = y_of(tick / 100)
        parts.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{tick}%</text>'
        )

    for color_index, color in enumerate(COLORS):
        group_x = left + color_index * group_width
        start = group_x + (group_width - bars_width) / 2
        for bar_index, (_, key) in enumerate(series):
            rate = by_color[key][color]
            x = start + bar_index * (bar_width + gap)
            y = y_of(rate)
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" '
                f'height="{max(top + plot_height - y, 1):.1f}" rx="3" fill="{SERIES[key]}"/>'
            )
            parts.append(
                f'<text class="tick" x="{x + bar_width / 2:.1f}" y="{y - 6:.1f}" '
                f'text-anchor="middle">{rate * 100:.0f}</text>'
            )
        label_y = top + plot_height + 26
        parts.append(
            f'<circle cx="{group_x + group_width / 2 - 34:.1f}" cy="{label_y - 4:.1f}" '
            f'r="6" fill="{PUZZLE_COLORS[color]}"/>'
        )
        parts.append(
            f'<text class="label" x="{group_x + group_width / 2 - 22:.1f}" y="{label_y:.1f}">'
            f"{color.title()}</text>"
        )

    parts.append(
        f'<line class="axis" x1="{left}" y1="{top + plot_height:.1f}" '
        f'x2="{left + plot_width}" y2="{top + plot_height:.1f}"/>'
    )

    legend_y = top + plot_height + 66
    x = left
    for name, key in series:
        parts.append(
            f'<rect x="{x:.1f}" y="{legend_y - 10:.1f}" width="12" height="12" rx="2" '
            f'fill="{SERIES[key]}"/>'
        )
        parts.append(f'<text class="label" x="{x + 19:.1f}" y="{legend_y:.1f}">{esc(name)}</text>')
        x += 24 + len(name) * 7.6
    parts.append(
        '<text class="sub" x="24" y="382">Network and language-model bars average 3 runs each; '
        "the rule baselines are single deterministic runs.</text>"
    )
    path.write_text(svg(width, height, "\n".join(parts)), encoding="utf-8")


def chart_rl_passes(data: dict, path: Path) -> None:
    """Lines: validation wins after each extra pass of RL over the training set."""
    seeds = data["repeated"]["seeds"]
    passes = [stage["rl_epochs"] for stage in seeds[0]["stages"]]
    seed_lines = [(seed["seed"], [stage["wins"] for stage in seed["stages"]]) for seed in seeds]
    means = [mean(wins[index] for _, wins in seed_lines) for index in range(len(passes))]

    width, height = 760, 372
    left, top = 64, 86
    plot_width, plot_height = 620, 170
    max_value = 30

    def x_of(index: int) -> float:
        return left + plot_width * index / (len(passes) - 1)

    def y_of(value: float) -> float:
        return top + plot_height * (1 - value / max_value)

    parts = [
        '<text class="title" x="24" y="34">Validation wins after each extra RL pass</text>',
        '<text class="sub" x="24" y="56">Three seeds, resumed from the same supervised '
        "checkpoints, replaying all 476 training puzzles per pass.</text>",
    ]

    for tick in range(0, max_value + 1, 10):
        y = y_of(tick)
        parts.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{tick}</text>'
        )

    for index, count in enumerate(passes):
        parts.append(
            f'<text class="label" x="{x_of(index):.1f}" y="{top + plot_height + 26:.1f}" '
            f'text-anchor="middle">{count}</text>'
        )
    parts.append(
        f'<text class="sub" x="{left + plot_width / 2:.1f}" y="{top + plot_height + 48:.1f}" '
        f'text-anchor="middle">RL passes over the training set</text>'
    )

    for seed, wins in seed_lines:
        points = " ".join(f"{x_of(i):.1f},{y_of(value):.1f}" for i, value in enumerate(wins))
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{SERIES["network"]}" '
            f'stroke-width="1.6" stroke-opacity="0.45"/>'
        )
        for i, value in enumerate(wins):
            parts.append(
                f'<circle cx="{x_of(i):.1f}" cy="{y_of(value):.1f}" r="3" '
                f'fill="{SERIES["network"]}" fill-opacity="0.55"/>'
            )

    points = " ".join(f"{x_of(i):.1f},{y_of(value):.1f}" for i, value in enumerate(means))
    parts.append(
        f'<polyline points="{points}" fill="none" stroke="{SERIES["llm"]}" stroke-width="2.8"/>'
    )
    for i, value in enumerate(means):
        parts.append(
            f'<circle cx="{x_of(i):.1f}" cy="{y_of(value):.1f}" r="4.5" fill="{SERIES["llm"]}"/>'
        )
        parts.append(
            f'<text class="value" x="{x_of(i):.1f}" y="{y_of(value) - 12:.1f}" '
            f'text-anchor="middle">{value:.1f}</text>'
        )

    parts.append(
        f'<line class="axis" x1="{left}" y1="{top - 6}" x2="{left}" y2="{top + plot_height:.1f}"/>'
    )
    parts.append(
        f'<line class="axis" x1="{left}" y1="{top + plot_height:.1f}" '
        f'x2="{left + plot_width}" y2="{top + plot_height:.1f}"/>'
    )
    parts.append(
        f'<rect x="{left}" y="{top + plot_height + 64:.1f}" width="12" height="12" rx="2" '
        f'fill="{SERIES["llm"]}"/>'
    )
    parts.append(
        f'<text class="label" x="{left + 19:.1f}" y="{top + plot_height + 74:.1f}">'
        "Mean</text>"
    )
    seed_legend_x = left + 84
    parts.append(
        f'<rect x="{seed_legend_x:.1f}" y="{top + plot_height + 64:.1f}" width="12" height="12" '
        f'rx="2" fill="{SERIES["network"]}" fill-opacity="0.55"/>'
    )
    seed_names = ", ".join(str(seed) for seed, _ in seed_lines)
    parts.append(
        f'<text class="label" x="{seed_legend_x + 19:.1f}" y="{top + plot_height + 74:.1f}">'
        f"Seeds {esc(seed_names)}</text>"
    )
    parts.append(
        '<text class="sub" x="24" y="356">The second pass helped on average; the third did not. '
        "Individual puzzles moved in both directions.</text>"
    )
    path.write_text(svg(width, height, "\n".join(parts)), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ASSETS)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = gather()
    charts = {
        "wins-by-agent.svg": chart_wins,
        "solve-rate-by-color.svg": chart_by_color,
        "rl-passes.svg": chart_rl_passes,
    }
    for name, draw in charts.items():
        path = args.output_dir / name
        draw(data, path)
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
