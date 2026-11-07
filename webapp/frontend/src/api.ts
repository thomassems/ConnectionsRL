import type {
  Puzzle,
  PuzzleSummary,
  RunOptions,
  RunResponse,
  StoredSolve,
} from "./types";

async function getJSON<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.error || `request failed (${response.status})`);
  }
  return response.json();
}

export async function fetchPuzzles(): Promise<PuzzleSummary[]> {
  const data = await getJSON<{ puzzles: PuzzleSummary[] }>("/api/puzzles");
  return data.puzzles;
}

export function fetchPuzzle(puzzleId: string): Promise<Puzzle> {
  return getJSON<Puzzle>(`/api/puzzles/${puzzleId}`);
}

export async function fetchStoredSolves(puzzleId: string): Promise<StoredSolve[]> {
  const data = await getJSON<{ solves: StoredSolve[] }>(
    `/api/puzzles/${puzzleId}/solves`,
  );
  return data.solves;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Start a live run and poll until it finishes. Runs are asynchronous on the
 * server (the reasoner can take minutes), so a single long request would be
 * dropped by the dev proxy — instead we kick off a job and poll its status.
 */
export interface RunProgress {
  guesses: number;
  solved: number;
  elapsedSec: number;
}

export async function runAgent(
  puzzleId: string,
  options: RunOptions,
  onProgress?: (p: RunProgress) => void,
): Promise<RunResponse> {
  const start = await fetch(`/api/puzzles/${puzzleId}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(options),
  });
  if (!start.ok) {
    const detail = await start.json().catch(() => ({}));
    throw new Error(detail.error || `run failed (${start.status})`);
  }
  const { job_id } = await start.json();
  const startedAt = Date.now();

  // Poll for up to ~10 minutes; the reasoner on a hard puzzle is the slow case.
  const deadline = startedAt + 10 * 60 * 1000;
  while (Date.now() < deadline) {
    await sleep(2000);
    const status = await getJSON<{
      status: string;
      progress?: { guesses: number; solved: number };
      result?: RunResponse["result"];
      meta?: RunResponse["meta"];
      error?: string;
    }>(`/api/runs/${job_id}`);
    onProgress?.({
      guesses: status.progress?.guesses ?? 0,
      solved: status.progress?.solved ?? 0,
      elapsedSec: Math.round((Date.now() - startedAt) / 1000),
    });
    if (status.status === "done" && status.result && status.meta) {
      return { result: status.result, meta: status.meta };
    }
    if (status.status === "error") {
      throw new Error(status.error || "run failed");
    }
  }
  throw new Error("run timed out");
}
