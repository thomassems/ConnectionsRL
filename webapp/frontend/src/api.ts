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

export async function runAgent(
  puzzleId: string,
  options: RunOptions,
): Promise<RunResponse> {
  const response = await fetch(`/api/puzzles/${puzzleId}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(options),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.error || `run failed (${response.status})`);
  }
  return response.json();
}
