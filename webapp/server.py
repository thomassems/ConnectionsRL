"""A small dependency-free HTTP API for the Connections replay web app.

It serves three things to the React frontend:

* the catalogue of saved puzzles (from ``data/puzzles/nyt/manifest.json``),
* a single puzzle's board and answer key (this is a private local tool, so
  exposing the key is fine; the frontend needs it to colour solved groups), and
* the agent's attempts: any *stored* solves found in ``reports/*.json`` and,
  on demand, a *live* run of the LLM solver against a chosen puzzle.

The server uses only the standard library so it adds no project dependencies.
A live run additionally needs the optional ``anthropic`` package and an API
key (``ANTHROPIC_API_KEY`` or ``DEEPSEEK_API_KEY``, read from the environment
or a local ``.env`` file); stored solves and browsing work without either.

Run it from the repository root so ``connections_rl`` imports resolve::

    .venv/bin/python -m webapp.server --port 8000

Then start the Vite dev server in ``webapp/frontend`` (``npm run dev``); its
proxy forwards ``/api`` here.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import traceback
from dataclasses import dataclass
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from connections_rl.llm_solver import (
    PROVIDERS,
    AnthropicChat,
    LLMSolver,
    estimate_cost,
    load_env_file,
    play_puzzle,
)
from connections_rl.puzzle import load_puzzle

REPO_ROOT = Path(__file__).resolve().parent.parent
PUZZLE_DIR = REPO_ROOT / "data" / "puzzles" / "nyt"
MANIFEST = PUZZLE_DIR / "manifest.json"
REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_SEED = 7

# One live run at a time. Running the solver takes many seconds and costs
# money, so a global lock keeps a trigger-happy UI from launching several at
# once against the same (or different) puzzles.
_run_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Puzzle catalogue and boards
# --------------------------------------------------------------------------- #
def board_order(puzzle, seed: int) -> list[str]:
    """Return the 16 words in the same shuffled order the environment uses.

    ``ConnectionsEnv.reset`` seeds ``random.Random(seed)`` and shuffles the
    flattened word list, so reproducing it here lines the displayed grid up
    with what the agent saw for that seed.
    """
    words = [word for group in puzzle.groups for word in group.words]
    random.Random(seed).shuffle(words)
    return words


def puzzle_payload(puzzle_id: str, seed: int) -> dict:
    puzzle = load_puzzle(PUZZLE_DIR / f"{puzzle_id}.json")
    entry = _manifest_index().get(puzzle_id, {})
    return {
        "puzzle_id": puzzle.puzzle_id,
        "date": entry.get("date"),
        "nyt_id": entry.get("nyt_id"),
        "seed": seed,
        "board": board_order(puzzle, seed),
        "groups": [
            {"title": group.title, "color": group.color, "words": list(group.words)}
            for group in puzzle.groups
        ],
    }


@lru_cache(maxsize=1)
def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _manifest_index() -> dict[str, dict]:
    index: dict[str, dict] = {}
    for entry in _manifest().get("puzzles", []):
        puzzle_id = f"nyt-{entry['date']}"
        index[puzzle_id] = {**entry, "puzzle_id": puzzle_id}
    return index


def catalogue() -> list[dict]:
    """Every saved puzzle, newest first, flagged if a stored solve exists."""
    stored = _stored_index()
    items = []
    for puzzle_id, entry in _manifest_index().items():
        if not (PUZZLE_DIR / f"{puzzle_id}.json").exists():
            continue
        items.append({
            "puzzle_id": puzzle_id,
            "date": entry.get("date"),
            "nyt_id": entry.get("nyt_id"),
            "stored_solves": len(stored.get(puzzle_id, [])),
        })
    items.sort(key=lambda item: item["date"] or "", reverse=True)
    return items


# --------------------------------------------------------------------------- #
# Stored solves from reports/
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StoredSolve:
    source: str
    provider: str | None
    model: str | None
    effort: str | None
    one_away_hint: bool | None
    seed: int | None
    result: dict

    def meta(self, index: int) -> dict:
        label = self.model or self.source
        if self.one_away_hint is not None:
            label += " · hint on" if self.one_away_hint else " · hint off"
        return {
            "id": f"{self.source}#{index}",
            "label": label,
            "source": self.source,
            "provider": self.provider,
            "model": self.model,
            "effort": self.effort,
            "one_away_hint": self.one_away_hint,
            "seed": self.seed,
            "status": self.result.get("status"),
            "groups_solved": self.result.get("groups_solved"),
            "mistakes": self.result.get("mistakes"),
            "turns": self.result.get("turns"),
        }


@lru_cache(maxsize=1)
def _stored_index() -> dict[str, list[StoredSolve]]:
    """Map each puzzle_id to the solves found across all LLM reports."""
    index: dict[str, list[StoredSolve]] = {}
    if not REPORTS_DIR.exists():
        return index
    for path in sorted(REPORTS_DIR.glob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(report, dict) or report.get("agent") != "llm":
            continue
        results = report.get("results")
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict) or "puzzle_id" not in result:
                continue
            solve = StoredSolve(
                source=path.stem,
                provider=report.get("provider"),
                model=report.get("model"),
                effort=report.get("effort"),
                one_away_hint=report.get("one_away_hint"),
                seed=report.get("seed"),
                result=result,
            )
            index.setdefault(result["puzzle_id"], []).append(solve)
    return index


def stored_solves(puzzle_id: str) -> list[dict]:
    solves = _stored_index().get(puzzle_id, [])
    return [
        {**solve.meta(index), "result": solve.result}
        for index, solve in enumerate(solves)
    ]


# --------------------------------------------------------------------------- #
# Live agent runs
# --------------------------------------------------------------------------- #
def run_live(puzzle_id: str, options: dict) -> dict:
    """Play one puzzle with the LLM solver and return the per-puzzle result.

    ``options`` mirrors the CLI flags: ``provider``, ``model``, ``effort``,
    ``one_away_hint``, ``seed``. The chosen provider's API key must be present
    in the environment or ``.env``.
    """
    provider_name = options.get("provider", "anthropic")
    if provider_name not in PROVIDERS:
        raise ValueError(f"unknown provider {provider_name!r}")
    provider = PROVIDERS[provider_name]
    model = options.get("model") or provider["default_model"]
    effort = options.get("effort", "medium") if provider["supports_effort"] else None
    seed = int(options.get("seed", DEFAULT_SEED))
    one_away_hint = bool(options.get("one_away_hint", True))

    load_env_file(REPO_ROOT / ".env")
    api_key = os.environ.get(provider["api_key_env"])
    if not api_key:
        raise PermissionError(
            f"set {provider['api_key_env']} (in the environment or .env) "
            f"to run the {provider_name} agent"
        )

    path = PUZZLE_DIR / f"{puzzle_id}.json"
    if not path.exists():
        raise FileNotFoundError(puzzle_id)

    chat = AnthropicChat(
        model=model,
        max_tokens=int(options.get("max_tokens", 2048)),
        effort=effort,
        base_url=provider["base_url"],
        api_key=api_key,
    )
    solver = LLMSolver(chat, one_away_hint=one_away_hint)
    result = play_puzzle(path, solver, seed)
    return {
        "result": result,
        "meta": {
            "id": f"live#{provider_name}:{model}",
            "label": f"{model} · live",
            "source": "live",
            "provider": provider_name,
            "model": model,
            "effort": effort,
            "one_away_hint": one_away_hint,
            "seed": seed,
            "status": result["status"],
            "groups_solved": result["groups_solved"],
            "mistakes": result["mistakes"],
            "turns": result["turns"],
            "input_tokens": solver.input_tokens,
            "output_tokens": solver.output_tokens,
            "estimated_cost_usd": estimate_cost(
                model, solver.input_tokens, solver.output_tokens
            ),
        },
    }


# --------------------------------------------------------------------------- #
# HTTP plumbing
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "ConnectionsReplay/1.0"

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # The Vite dev server proxies /api, so same-origin in practice; the
        # permissive CORS header just makes direct calls during debugging work.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._send(200, {})

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/api/health":
                self._send(200, {"ok": True})
            elif path == "/api/puzzles":
                self._send(200, {"puzzles": catalogue()})
            elif path.startswith("/api/puzzles/") and path.endswith("/solves"):
                puzzle_id = path[len("/api/puzzles/"):-len("/solves")]
                self._send(200, {"solves": stored_solves(puzzle_id)})
            elif path.startswith("/api/puzzles/"):
                puzzle_id = path[len("/api/puzzles/"):]
                self._send(200, puzzle_payload(puzzle_id, DEFAULT_SEED))
            else:
                self._send(404, {"error": "not found"})
        except FileNotFoundError:
            self._send(404, {"error": "puzzle not found"})
        except Exception as exc:  # pragma: no cover - defensive
            traceback.print_exc()
            self._send(500, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if not (path.startswith("/api/puzzles/") and path.endswith("/run")):
            self._send(404, {"error": "not found"})
            return
        puzzle_id = path[len("/api/puzzles/"):-len("/run")]
        options = self._read_json()
        if not _run_lock.acquire(blocking=False):
            self._send(429, {"error": "an agent run is already in progress"})
            return
        try:
            self._send(200, run_live(puzzle_id, options))
        except PermissionError as exc:
            self._send(400, {"error": str(exc), "needs_api_key": True})
        except FileNotFoundError:
            self._send(404, {"error": "puzzle not found"})
        except Exception as exc:  # pragma: no cover - defensive
            traceback.print_exc()
            self._send(500, {"error": str(exc)})
        finally:
            _run_lock.release()

    def log_message(self, fmt: str, *args) -> None:
        # Quieter than the default; one concise line per request.
        print(f"[api] {self.address_string()} {fmt % args}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    # Fail fast with a clear message if run from the wrong directory.
    if not MANIFEST.exists():
        parser.error(f"manifest not found at {MANIFEST}; run from the repo root")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    stored = sum(len(v) for v in _stored_index().values())
    print(
        f"Connections replay API on http://{args.host}:{args.port}\n"
        f"  {len(catalogue())} puzzles, {stored} stored solves indexed\n"
        f"  frontend: cd webapp/frontend && npm install && npm run dev"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
