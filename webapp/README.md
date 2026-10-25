# Connections Agent Replay (local web app)

A private local tool to browse the saved puzzles and watch the LLM agent solve
them — either replaying a run already saved in `reports/`, or running the agent
live against a puzzle you pick.

It has two parts:

- **`server.py`** — a dependency-free Python HTTP API (standard library only)
  that reads puzzles from `data/puzzles/nyt/`, stored solves from `reports/`,
  and can run `connections_rl.llm_solver` on demand.
- **`frontend/`** — a Vite + React + TypeScript single-page app that draws the
  board, animates each guess, and lets you scrub through the solve.

## Run it

Two terminals, both from the repository root.

**1. API** (uses the project venv):

```sh
.venv/bin/python -m webapp.server --port 8000
```

**2. Frontend** (first time installs deps):

```sh
cd webapp/frontend
npm install
npm run dev
```

Open the URL Vite prints (http://localhost:5173). The dev server proxies
`/api` to the Python server on port 8000, so both run as one origin in the
browser.

## Using it

- **Refresh puzzle** picks a random saved puzzle; the dropdown jumps to a
  specific date. The app starts on a random puzzle that has a stored solve.
- **Stored solves** lists every saved run for that puzzle found in `reports/`
  (there are solves for the 141 validation-split puzzles). Click one to load
  its replay.
- **Play / scrub / speed** step through the guesses. Highlighted tiles are the
  current guess; the banner shows Correct / One away / Incorrect; solved groups
  drop in with their real colour and category.
- **Run the agent live** plays the chosen puzzle through the real solver. This
  calls the model API and costs tokens, so it needs the provider's key:
  `ANTHROPIC_API_KEY` or `DEEPSEEK_API_KEY`, in your environment or the repo's
  `.env` (same as the CLI). Only one live run happens at a time.

## API endpoints

| Method | Path | Returns |
| --- | --- | --- |
| GET | `/api/puzzles` | catalogue (date, NYT #, stored-solve count) |
| GET | `/api/puzzles/{id}` | board (seeded word order) + answer key |
| GET | `/api/puzzles/{id}/solves` | stored solves for that puzzle |
| POST | `/api/puzzles/{id}/run` | run the agent live; body mirrors the CLI flags |

The board word order reproduces `ConnectionsEnv.reset(seed=7)`, so the grid
matches what the agent saw.
