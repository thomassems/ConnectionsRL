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
from datetime import datetime
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
# Live runs triggered from the UI are saved here so they persist and show up
# in a puzzle's stored-solve list, kept apart from the curated benchmark reports.
LIVE_DIR = REPORTS_DIR / "live"
DEFAULT_SEED = 7

# One live run at a time. Running the solver takes many seconds (minutes for
# the reasoner) and costs money, so a global lock keeps a trigger-happy UI from
# launching several at once against the same (or different) puzzles.
_run_lock = threading.Lock()

# Live runs execute in a background thread so the HTTP request returns at once;
# the UI polls the job for the result. This avoids a minutes-long request being
# dropped by the dev proxy before the (slow) reasoner finishes. Each job is a
# dict: status "running"|"done"|"error", plus result/meta or error when done.
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_MAX_JOBS = 50


def _start_run(puzzle_id: str, options: dict) -> str | None:
    """Kick off a live run in a background thread; return its job id.

    Returns ``None`` if a run is already in progress (single-flight).
    """
    if not _run_lock.acquire(blocking=False):
        return None
    import uuid  # noqa: PLC0415 - stdlib, imported lazily

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        # Keep the registry from growing without bound in long sessions.
        if len(_jobs) >= _MAX_JOBS:
            for stale in [k for k, v in _jobs.items() if v["status"] != "running"][:10]:
                _jobs.pop(stale, None)
        _jobs[job_id] = {"status": "running", "puzzle_id": puzzle_id,
                         "progress": {"guesses": 0, "solved": 0}}

    def on_progress(guesses: int, solved: int) -> None:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job["progress"] = {"guesses": guesses, "solved": solved}

    def worker() -> None:
        update: dict = {}
        try:
            out = run_live(puzzle_id, options, on_progress=on_progress)
            update = {"status": "done", "result": out["result"], "meta": out["meta"]}
        except PermissionError as exc:
            update = {"status": "error", "error": str(exc), "needs_api_key": True}
        except FileNotFoundError:
            update = {"status": "error", "error": "puzzle not found"}
        except Exception as exc:  # pragma: no cover - defensive
            traceback.print_exc()
            update = {"status": "error", "error": str(exc)}
        finally:
            with _jobs_lock:
                _jobs[job_id].update(update)
            _run_lock.release()

    threading.Thread(target=worker, daemon=True).start()
    return job_id


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
    temperature: float | None = None
    live: bool = False

    def meta(self, index: int) -> dict:
        label = self.model or self.source
        if self.live:
            label += " · live"
        if self.temperature is not None:
            label += f" · temp {self.temperature:g}"
        if self.one_away_hint is not None:
            label += " · hint on" if self.one_away_hint else " · hint off"
        return {
            "id": f"{self.source}#{index}",
            "label": label,
            "source": self.source,
            "provider": self.provider,
            "model": self.model,
            "effort": self.effort,
            "temperature": self.temperature,
            "one_away_hint": self.one_away_hint,
            "seed": self.seed,
            "live": self.live,
            "status": self.result.get("status"),
            "groups_solved": self.result.get("groups_solved"),
            "mistakes": self.result.get("mistakes"),
            "turns": self.result.get("turns"),
        }


def _report_paths() -> list[Path]:
    """Curated reports first, then saved live runs."""
    paths = sorted(REPORTS_DIR.glob("*.json")) if REPORTS_DIR.exists() else []
    if LIVE_DIR.exists():
        paths += sorted(LIVE_DIR.glob("*.json"))
    return paths


@lru_cache(maxsize=1)
def _stored_index() -> dict[str, list[StoredSolve]]:
    """Map each puzzle_id to the solves found across all LLM reports."""
    index: dict[str, list[StoredSolve]] = {}
    for path in _report_paths():
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
                temperature=report.get("temperature"),
                live=report.get("source_kind") == "live",
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
def _save_live_run(puzzle_id: str, report: dict) -> str:
    """Write one live run as its own report and return its file stem.

    The file is shaped like the curated reports so ``_stored_index`` picks it
    up; ``source_kind`` marks it as a live run. Microsecond timestamps keep
    rapid successive runs from colliding.
    """
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{puzzle_id}-live-{datetime.now():%Y%m%d-%H%M%S-%f}"
    (LIVE_DIR / f"{stem}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return stem


def run_live(puzzle_id: str, options: dict, on_progress=None) -> dict:
    """Play one puzzle with the LLM solver, save it, and return the solve.

    ``options`` mirrors the CLI flags: ``provider``, ``model``, ``effort``,
    ``temperature``, ``one_away_hint``, ``seed``. The chosen provider's API key
    must be present in the environment or ``.env``. The finished run is saved so
    it joins the puzzle's stored solves; the return value matches a
    ``stored_solves`` entry (so the UI can select it from the refreshed list).
    """
    provider_name = options.get("provider") or "deepseek"
    if provider_name not in PROVIDERS:
        raise ValueError(f"unknown provider {provider_name!r}")
    provider = PROVIDERS[provider_name]
    model = options.get("model") or provider["default_model"]
    effort = options.get("effort", "medium") if provider["supports_effort"] else None
    seed = int(options.get("seed", DEFAULT_SEED))
    one_away_hint = bool(options.get("one_away_hint", True))
    temperature = options.get("temperature")
    if temperature is not None:
        temperature = max(0.0, min(1.0, float(temperature)))

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
        temperature=temperature,
        base_url=provider["base_url"],
        api_key=api_key,
    )
    solver = LLMSolver(chat, one_away_hint=one_away_hint)
    result = play_puzzle(path, solver, seed, on_guess=on_progress)

    report = {
        "agent": "llm",
        "provider": provider_name,
        "model": model,
        "effort": effort,
        "temperature": temperature,
        "one_away_hint": one_away_hint,
        "seed": seed,
        "source_kind": "live",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_tokens": solver.input_tokens,
        "output_tokens": solver.output_tokens,
        "estimated_cost_usd": estimate_cost(
            model, solver.input_tokens, solver.output_tokens
        ),
        "results": [result],
    }
    stem = _save_live_run(puzzle_id, report)
    _stored_index.cache_clear()  # so the new run appears in stored solves

    # Return it in the same shape as a stored solve, with cost/token extras.
    for entry in stored_solves(puzzle_id):
        if entry["source"] == stem:
            saved_result = entry.pop("result")
            entry.update({
                "input_tokens": report["input_tokens"],
                "output_tokens": report["output_tokens"],
                "estimated_cost_usd": report["estimated_cost_usd"],
            })
            return {"result": saved_result, "meta": entry}
    # Fallback: should not happen, but never fail the request over it.
    return {"result": result, "meta": {"id": f"{stem}#0", "label": f"{model} · live"}}


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
            elif path.startswith("/api/runs/"):
                job_id = path[len("/api/runs/"):]
                with _jobs_lock:
                    job = _jobs.get(job_id)
                if job is None:
                    self._send(404, {"error": "unknown run"})
                else:
                    self._send(200, {k: v for k, v in job.items() if k != "puzzle_id"})
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
        if not (PUZZLE_DIR / f"{puzzle_id}.json").exists():
            self._send(404, {"error": "puzzle not found"})
            return
        options = self._read_json()
        job_id = _start_run(puzzle_id, options)
        if job_id is None:
            self._send(409, {"error": "an agent run is already in progress"})
        else:
            # 202 Accepted: the run is underway; poll GET /api/runs/{job_id}.
            self._send(202, {"job_id": job_id, "status": "running"})

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
