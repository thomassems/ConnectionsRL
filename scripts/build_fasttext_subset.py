"""Build a compact SQLite cache from official pretrained fastText .vec.gz data."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sqlite3
import struct
import tempfile
from pathlib import Path

from connections_rl.puzzle import load_puzzle


SOURCE_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz"
DEFINITION_WORDS = re.compile(r"[a-z]+(?:'[a-z]+)?")


def collect_terms(puzzles: Path, reader: object) -> tuple[set[str], set[str]]:
    paths = sorted(path for path in puzzles.glob("*.json") if path.name != "manifest.json")
    if not paths:
        raise ValueError(f"no puzzle files found in {puzzles}")
    groups = [group for path in paths for group in load_puzzle(path).groups]
    displayed = {word for group in groups for word in group.words}
    terms = {token for word in displayed for token in word.casefold().split()}
    terms.update(token for group in groups for token in group.title.casefold().split())
    for word in sorted(displayed):
        normalized = word.casefold().replace(" ", "_")
        for pos in ("n", "v", "a", "s", "r"):
            for lemma in reader.lemmas(normalized, pos=pos):
                terms.update(DEFINITION_WORDS.findall(
                    lemma.synset().definition().casefold()
                ))
    return terms, displayed


def build(source: Path, output: Path, terms: set[str]) -> dict:
    if output.exists():
        raise FileExistsError(f"vector cache already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as stream:
        source_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    target_bytes = {token.encode("utf-8"): token for token in terms}
    temporary = tempfile.NamedTemporaryFile(
        prefix=output.name + ".", suffix=".tmp", dir=output.parent, delete=False
    )
    temporary.close()
    try:
        connection = sqlite3.connect(temporary.name)
        connection.execute("CREATE TABLE vectors (word TEXT PRIMARY KEY, data BLOB NOT NULL)")
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        found = 0
        with gzip.open(source, "rb") as stream:
            header = stream.readline().split()
            if len(header) != 2:
                raise ValueError("fastText text file has no vocabulary/dimension header")
            declared_count, dimension = (int(value) for value in header)
            for index, line in enumerate(stream, 1):
                token_bytes, separator, values_bytes = line.partition(b" ")
                if separator and token_bytes in target_bytes:
                    values = tuple(float(value) for value in values_bytes.split())
                    if len(values) != dimension:
                        raise ValueError(f"bad vector length on line {index}")
                    connection.execute(
                        "INSERT OR IGNORE INTO vectors (word, data) VALUES (?, ?)",
                        (target_bytes[token_bytes], struct.pack(f"<{dimension}f", *values)),
                    )
                    found += 1
                if index % 250_000 == 0:
                    print(f"Scanned {index:,}/{declared_count:,} vectors; matched {found:,}",
                          flush=True)
        metadata = {
            "source_url": SOURCE_URL,
            "source_sha256": source_sha256,
            "dimension": str(dimension),
            "requested_terms": str(len(terms)),
            "matched_terms": str(found),
            "declared_source_terms": str(declared_count),
        }
        connection.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)", metadata.items()
        )
        connection.commit()
        connection.close()
        os.replace(temporary.name, output)
        return metadata
    finally:
        if os.path.exists(temporary.name):
            os.unlink(temporary.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--puzzles", type=Path, default=Path("data/puzzles/nyt"))
    parser.add_argument("--output", type=Path,
                        default=Path(".venv/models/cc.en.300.subset.sqlite3"))
    parser.add_argument("--wordnet-data", type=Path)
    args = parser.parse_args()

    import nltk
    local_data = Path(__file__).resolve().parent.parent / ".venv/nltk_data"
    data_dir = args.wordnet_data or (local_data if local_data.exists() else None)
    if data_dir is not None:
        nltk.data.path.insert(0, str(data_dir.resolve()))
    from nltk.corpus import wordnet

    terms, displayed = collect_terms(args.puzzles, wordnet)
    print(f"Requested {len(terms):,} terms from {len(displayed):,} distinct puzzle words",
          flush=True)
    metadata = build(args.source, args.output, terms)
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"Saved compact vectors to {args.output}", flush=True)


if __name__ == "__main__":
    main()
