"""Create reproducible chronological train/validation/test puzzle splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from connections_rl.puzzle import load_puzzle


def make_splits(manifest_path: Path, train_end: date, validation_end: date) -> dict:
    if train_end >= validation_end:
        raise ValueError("train_end must be before validation_end")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    splits: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    seen: set[str] = set()
    for record in manifest["puzzles"]:
        day = date.fromisoformat(record["date"])
        puzzle_id = f"nyt-{day.isoformat()}"
        if puzzle_id in seen:
            raise ValueError(f"duplicate puzzle: {puzzle_id}")
        seen.add(puzzle_id)
        path = manifest_path.parent / f"{puzzle_id}.json"
        if load_puzzle(path).puzzle_id != puzzle_id:
            raise ValueError(f"puzzle id does not match manifest: {path}")
        key = "train" if day <= train_end else "validation" if day <= validation_end else "test"
        splits[key].append(puzzle_id)
    if len(seen) != manifest["count"] or any(not ids for ids in splits.values()):
        raise ValueError("manifest count differs or a split is empty")
    return {
        "source": "NYT Connections dated JSON",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "train_through": train_end.isoformat(),
        "validation_through": validation_end.isoformat(),
        "splits": {key: sorted(ids) for key, ids in splits.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/puzzles/nyt/manifest.json"))
    parser.add_argument("--train-through", type=date.fromisoformat, default=date(2024, 9, 30))
    parser.add_argument("--validation-through", type=date.fromisoformat, default=date(2025, 2, 19))
    parser.add_argument("--output", type=Path, default=Path("data/splits/nyt-v1.json"))
    args = parser.parse_args()
    result = make_splits(args.manifest, args.train_through, args.validation_through)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Saved " + ", ".join(f"{key}={len(ids)}" for key, ids in result["splits"].items()))


if __name__ == "__main__":
    main()
