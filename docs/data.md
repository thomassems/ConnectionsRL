# Puzzle data

[← back to the README](../README.md)

## The archive

`data/puzzles/nyt/` contains **717 dated historical puzzles**, fetched from the [NYT Connections JSON endpoint](https://www.nytimes.com/svc/connections/v2/2024-01-01.json) and spanning June 12, 2023 – May 31, 2025.

Three visual-card dates (April 1 and December 12, 2024, and April 1, 2025) were excluded, because this agent takes words as input and those puzzles are pictures.

The endpoint lists the four categories in color order but does not label the colors; the importer assigns yellow, green, blue and purple in that order. `manifest.json` records dates, NYT puzzle numbers, source response hashes and the exclusions.

Cross-checking the 717 saved puzzles against the [Eyefyre answer archive](https://github.com/Eyefyre/NYT-Connections-Answers) found 714 matches and three dates whose words differ: July 19 and December 13, 2023, and September 20, 2024. The saved records use the current NYT responses as the source of record.

## The split

`data/splits/nyt-v1.json` fixes a chronological split:

| Split | Count | Dates |
| --- | ---: | --- |
| Train | 476 | through September 30, 2024 |
| Validation | 141 | October 1, 2024 – February 19, 2025 |
| Test | 100 | February 20 – May 31, 2025 |

The split file lists exact puzzle IDs and a manifest hash. **The test period has not been used by anything reported in this repository**: not by the baselines, not for training, not for evaluation. It stays sealed until a final test is run deliberately.

Keep synthetic or generated puzzles in a separate training-only source, so test scores keep measuring performance on real, unseen puzzles.

## Record format

Each JSON record has a unique `puzzle_id` and exactly four groups. Each group has a category `title`, one of the colors `yellow`, `green`, `blue` or `purple`, and exactly four `words`. All sixteen words must be distinct.

```json
{
  "puzzle_id": "example-001",
  "groups": [
    {"title": "Colors", "color": "yellow", "words": ["RED", "BLUE", "GREEN", "GOLD"]},
    {"title": "Animals", "color": "green", "words": ["DOG", "CAT", "HORSE", "SHEEP"]},
    {"title": "Fruits", "color": "blue", "words": ["APPLE", "PEAR", "BANANA", "PEACH"]},
    {"title": "Shapes", "color": "purple", "words": ["CIRCLE", "SQUARE", "TRIANGLE", "OVAL"]}
  ]
}
```

`connections_rl.load_puzzle` reads this into the `Puzzle` and `Group` types. The loader converts JSON arrays to tuples, so Python constructing these types directly must also use tuples. The `puzzle_id` is kept for tracking and is never part of the agent's observation.

`data/puzzles/example.json` is a synthetic, deliberately simple puzzle used by the demo and the tests. It is not a historical NYT puzzle.

## Fetching more

`scripts/ingest_nyt.py` fetches other dated windows with `--start YYYY-MM-DD --days N`. It skips dates already in the manifest, validates each new puzzle, and preserves prior records when extending the collection. `--archive path/to/connections.json` adds a color-order cross-check against a local answer archive, though that check can reject dates where the archive and the current NYT response disagree.

Regenerate the split after changing the collection:

```sh
.venv/bin/python scripts/make_splits.py
```

Both the endpoint and the archive are external sources, so reruns may differ if either changes upstream.
