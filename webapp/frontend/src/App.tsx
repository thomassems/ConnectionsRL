import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchPuzzle,
  fetchPuzzles,
  fetchStoredSolves,
  runAgent,
} from "./api";
import type {
  Puzzle,
  PuzzleSummary,
  RunOptions,
  SolveMeta,
  SolveResult,
  StoredSolve,
} from "./types";
import { stateAfter } from "./replay";
import { usePlayback } from "./usePlayback";
import { Board } from "./components/Board";
import { Controls } from "./components/Controls";
import { GuessLog } from "./components/GuessLog";
import { SolvePanel } from "./components/SolvePanel";

interface ActiveSolve {
  meta: SolveMeta;
  result: SolveResult;
}

export default function App() {
  const [puzzles, setPuzzles] = useState<PuzzleSummary[]>([]);
  const [puzzle, setPuzzle] = useState<Puzzle | null>(null);
  const [stored, setStored] = useState<StoredSolve[]>([]);
  const [active, setActive] = useState<ActiveSolve | null>(null);
  const [speed, setSpeed] = useState(1);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const guesses = active?.result.guesses ?? [];
  const playback = usePlayback(guesses, speed);
  const { cursor, activeGuess, seek } = playback;

  const boardState = useMemo(
    () => (puzzle ? stateAfter(puzzle, guesses, cursor) : null),
    [puzzle, guesses, cursor],
  );

  // Load the catalogue once, then show a random puzzle to start.
  useEffect(() => {
    fetchPuzzles()
      .then((list) => {
        setPuzzles(list);
        if (list.length > 0) {
          const withSolves = list.filter((p) => p.stored_solves > 0);
          const pick = (withSolves.length ? withSolves : list)[
            Math.floor(Math.random() * (withSolves.length || list.length))
          ];
          void loadPuzzle(pick.puzzle_id);
        }
      })
      .catch((e) => setError(String(e.message || e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadPuzzle = useCallback(async (puzzleId: string) => {
    setError(null);
    setRunError(null);
    setActive(null);
    setStored([]);
    try {
      const [board, solves] = await Promise.all([
        fetchPuzzle(puzzleId),
        fetchStoredSolves(puzzleId),
      ]);
      setPuzzle(board);
      setStored(solves);
      if (solves.length > 0) {
        setActive({ meta: solves[0], result: solves[0].result });
      }
    } catch (e) {
      setError(String((e as Error).message || e));
    }
  }, []);

  const refreshPuzzle = useCallback(() => {
    if (puzzles.length === 0) return;
    const pick = puzzles[Math.floor(Math.random() * puzzles.length)];
    void loadPuzzle(pick.puzzle_id);
  }, [puzzles, loadPuzzle]);

  const onRun = useCallback(
    async (options: RunOptions) => {
      if (!puzzle) return;
      setRunning(true);
      setRunError(null);
      try {
        const response = await runAgent(puzzle.puzzle_id, options);
        setActive({ meta: response.meta, result: response.result });
      } catch (e) {
        setRunError(String((e as Error).message || e));
      } finally {
        setRunning(false);
      }
    },
    [puzzle],
  );

  const result = active?.result;
  const meta = active?.meta;

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Connections Agent Replay</h1>
          <p className="subtitle">Watch the solver work through a saved puzzle.</p>
        </div>
        <div className="picker">
          <select
            value={puzzle?.puzzle_id ?? ""}
            onChange={(e) => void loadPuzzle(e.target.value)}
          >
            {puzzles.map((p) => (
              <option key={p.puzzle_id} value={p.puzzle_id}>
                {p.date} · #{p.nyt_id}
                {p.stored_solves ? ` · ${p.stored_solves} solves` : ""}
              </option>
            ))}
          </select>
          <button className="primary" onClick={refreshPuzzle}>Refresh puzzle</button>
        </div>
      </header>

      {error && <div className="error banner">{error}</div>}

      {puzzle && boardState ? (
        <div className="layout">
          <main className="stage">
            <div className="puzzle-meta">
              <span className="pill">{puzzle.date}</span>
              <span className="pill">NYT #{puzzle.nyt_id}</span>
              {meta && <span className="pill accent">{meta.label}</span>}
            </div>

            <Board state={boardState} activeGuess={activeGuess} />

            {guesses.length > 0 ? (
              <Controls
                cursor={cursor}
                total={playback.total}
                playing={playback.playing}
                activeGuess={activeGuess}
                speed={speed}
                onPlay={playback.play}
                onPause={playback.pause}
                onStepBack={playback.stepBack}
                onStepForward={playback.stepForward}
                onRestart={playback.restart}
                onSeek={seek}
                onSpeed={setSpeed}
              />
            ) : (
              <p className="muted center">
                Pick a stored solve or run the agent to see a replay.
              </p>
            )}
          </main>

          <aside className="sidebar">
            {result && meta && (
              <div className="result-card">
                <div className={`result-status ${result.status}`}>
                  {result.status === "won" ? "Solved" : result.status === "lost" ? "Failed" : "—"}
                </div>
                <dl>
                  <div><dt>Groups</dt><dd>{result.groups_solved}/4</dd></div>
                  <div><dt>Mistakes</dt><dd>{result.mistakes}/4</dd></div>
                  <div><dt>Turns</dt><dd>{result.turns}</dd></div>
                  {meta.estimated_cost_usd != null && (
                    <div><dt>Est. cost</dt><dd>${meta.estimated_cost_usd.toFixed(4)}</dd></div>
                  )}
                </dl>
              </div>
            )}

            {guesses.length > 0 && (
              <GuessLog guesses={guesses} cursor={cursor} onSeek={seek} />
            )}

            <SolvePanel
              stored={stored}
              activeId={meta?.id ?? null}
              running={running}
              runError={runError}
              onSelectStored={(solve) => setActive({ meta: solve, result: solve.result })}
              onRun={onRun}
            />
          </aside>
        </div>
      ) : (
        !error && <p className="muted center">Loading…</p>
      )}
    </div>
  );
}
