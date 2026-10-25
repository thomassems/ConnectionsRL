export type Color = "yellow" | "green" | "blue" | "purple";
export type Feedback = "correct" | "one_away" | "incorrect";

export interface PuzzleSummary {
  puzzle_id: string;
  date: string | null;
  nyt_id: number | null;
  stored_solves: number;
}

export interface Group {
  title: string;
  color: Color;
  words: string[];
}

export interface Puzzle {
  puzzle_id: string;
  date: string | null;
  nyt_id: number | null;
  seed: number;
  board: string[];
  groups: Group[];
}

export interface Guess {
  words: string[];
  feedback: Feedback;
  reward: number;
  revealed_color: Color | null;
  attempts: number;
  fell_back: boolean;
}

export interface SolveResult {
  puzzle_id: string;
  status: "won" | "lost" | "in_progress";
  groups_solved: number;
  mistakes: number;
  turns: number;
  total_reward: number;
  guesses: Guess[];
}

export interface SolveMeta {
  id: string;
  label: string;
  source: string;
  provider: string | null;
  model: string | null;
  effort: string | null;
  one_away_hint: boolean | null;
  seed: number | null;
  status: string;
  groups_solved: number | null;
  mistakes: number | null;
  turns: number | null;
  input_tokens?: number;
  output_tokens?: number;
  estimated_cost_usd?: number | null;
}

export interface StoredSolve extends SolveMeta {
  result: SolveResult;
}

export interface RunResponse {
  result: SolveResult;
  meta: SolveMeta;
}

export interface RunOptions {
  provider: string;
  model?: string;
  effort?: string;
  one_away_hint: boolean;
  seed: number;
}
