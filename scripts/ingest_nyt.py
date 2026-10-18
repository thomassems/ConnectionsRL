"""Fetch and validate dated NYT Connections puzzles from the public JSON endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import date, timedelta
from pathlib import Path

from connections_rl.puzzle import Group, Puzzle, load_puzzle


COLORS = ("yellow", "green", "blue", "purple")
SOURCE = "https://www.nytimes.com/svc/connections/v2/{date}.json"


class UnsupportedPuzzle(ValueError):
    """A puzzle needs a different input modality than displayed words."""


def convert_official(payload: dict, expected_date: str) -> Puzzle:
    if payload.get("status") != "OK" or payload.get("print_date") != expected_date:
        raise ValueError(f"NYT response does not match {expected_date}")
    if not isinstance(payload.get("id"), int) or not isinstance(payload.get("categories"), list):
        raise ValueError(f"NYT response has no puzzle id or categories for {expected_date}")
    if len(payload["categories"]) != 4:
        raise ValueError(f"NYT response has {len(payload['categories'])} categories for {expected_date}")
    groups = []
    for color, category in zip(COLORS, payload["categories"]):
        if not isinstance(category, dict) or not isinstance(category.get("cards"), list):
            raise ValueError(f"malformed category on {expected_date}")
        if any("content" not in card for card in category["cards"]):
            raise UnsupportedPuzzle(f"non-text cards on {expected_date}")
        groups.append(Group(
            title=category["title"],
            color=color,
            words=tuple(card["content"] for card in category["cards"]),
        ))
    if all(not any(character.isalpha() for character in word)
           for group in groups for word in group.words):
        raise UnsupportedPuzzle(f"symbol-only cards on {expected_date}")
    return Puzzle(f"nyt-{expected_date}", tuple(groups))


def validate_archive(puzzle: Puzzle, archive: dict[str, dict]) -> None:
    day = puzzle.puzzle_id.removeprefix("nyt-")
    reference = archive.get(day)
    if reference is None:
        raise ValueError(f"archive has no matching date for {day}")
    answers = reference.get("answers", [])
    if len(answers) != 4 or sorted(answer.get("level") for answer in answers) != list(range(4)):
        raise ValueError(f"archive has no complete color labels for {day}")
    for level, group in enumerate(puzzle.groups):
        answer = next(answer for answer in answers if answer["level"] == level)
        if set(answer.get("members", [])) != set(group.words):
            raise ValueError(f"archive disagrees with NYT category order for {day}")


def fetch(date_text: str) -> tuple[dict, bytes]:
    url = SOURCE.format(date=date_text)
    result = subprocess.run(
        ["curl", "--fail", "--location", "--silent", "--show-error",
         "--retry", "2", "--max-time", "20", url],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"could not fetch {url}: {result.stderr.decode(errors='replace')}")
    return json.loads(result.stdout), result.stdout


def ingest(
    starts: list[date], days: int, output: Path, archive_path: Path | None,
    delay: float = 0.2,
) -> dict:
    if days < 1 or delay < 0:
        raise ValueError("days must be positive and delay must be nonnegative")
    archive = None
    archive_digest = None
    if archive_path is not None:
        archive_bytes = archive_path.read_bytes()
        archive = {item["date"]: item for item in json.loads(archive_bytes)}
        archive_digest = hashlib.sha256(archive_bytes).hexdigest()
    dates = sorted({start + timedelta(days=offset) for start in starts for offset in range(days)})
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if previous is not None and previous.get("source_url_template") != SOURCE:
        raise ValueError("existing manifest uses a different source")
    records = {item["date"]: item for item in previous["puzzles"]} if previous else {}
    skipped = {item["date"]: item for item in previous.get("skipped", [])} if previous else {}
    if previous is not None and len(records) != len(previous["puzzles"]):
        raise ValueError("existing manifest contains duplicate dates")
    for day, item in records.items():
        path = output / f"nyt-{day}.json"
        if not path.exists():
            raise FileNotFoundError(f"manifest puzzle is missing: {path}")
        load_puzzle(path)
    if archive_digest is None and previous is not None:
        archive_digest = previous.get("archive_check_sha256")
    elif previous is not None and previous.get("archive_check_sha256") not in (None, archive_digest):
        raise ValueError("archive checksum differs from existing manifest")

    def save_manifest() -> None:
        manifest = {
            "source_url_template": SOURCE,
            "color_order": list(COLORS),
            "archive_check_sha256": archive_digest,
            "count": len(records),
            "puzzles": [records[day] for day in sorted(records)],
            "skipped": [skipped[day] for day in sorted(skipped)],
        }
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)

    added = 0
    for index, day in enumerate(dates, 1):
        date_text = day.isoformat()
        if date_text in records or date_text in skipped:
            continue
        payload, raw = fetch(date_text)
        try:
            puzzle = convert_official(payload, date_text)
        except UnsupportedPuzzle as exc:
            skipped[date_text] = {"date": date_text, "reason": str(exc),
                                  "source_sha256": hashlib.sha256(raw).hexdigest()}
            save_manifest()
            print(f"Skipped {date_text}: {exc}", flush=True)
            continue
        if archive is not None:
            validate_archive(puzzle, archive)
        target = output / f"{puzzle.puzzle_id}.json"
        if target.exists():
            raise FileExistsError(f"untracked puzzle file already exists: {target}")
        serializable = {
            "puzzle_id": puzzle.puzzle_id,
            "groups": [
                {"title": group.title, "color": group.color, "words": list(group.words)}
                for group in puzzle.groups
            ],
        }
        target.write_text(json.dumps(serializable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        records[date_text] = {
            "date": date_text, "nyt_id": payload["id"],
            "source_sha256": hashlib.sha256(raw).hexdigest(),
        }
        added += 1
        save_manifest()
        if index % 10 == 0 or index == len(dates):
            print(f"Checked {index}/{len(dates)} dates; added {added}", flush=True)
        if index < len(dates):
            time.sleep(delay)
    save_manifest()
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="append", type=date.fromisoformat, required=True)
    parser.add_argument("--days", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/puzzles/nyt"))
    parser.add_argument("--archive", type=Path, help="optional colored archive for cross-checking")
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()
    manifest = ingest(args.start, args.days, args.output, args.archive, args.delay)
    print(f"Saved {manifest['count']} puzzles to {args.output}")


if __name__ == "__main__":
    main()
