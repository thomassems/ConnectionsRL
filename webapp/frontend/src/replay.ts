import type { Color, Guess, Puzzle } from "./types";

export interface SolvedGroup {
  color: Color;
  title: string;
  words: string[];
}

export interface ReplayState {
  solved: SolvedGroup[];
  remaining: string[]; // still on the board, in original board order
  mistakesUsed: number;
}

function groupByColor(puzzle: Puzzle): Map<Color, SolvedGroup> {
  const map = new Map<Color, SolvedGroup>();
  for (const group of puzzle.groups) {
    map.set(group.color, {
      color: group.color,
      title: group.title,
      words: group.words,
    });
  }
  return map;
}

/**
 * The board state after the first `n` guesses have fully resolved.
 * `n === 0` is the untouched board; `n === guesses.length` is the end.
 */
export function stateAfter(
  puzzle: Puzzle,
  guesses: Guess[],
  n: number,
): ReplayState {
  const byColor = groupByColor(puzzle);
  const solved: SolvedGroup[] = [];
  const solvedWords = new Set<string>();
  let mistakesUsed = 0;

  for (const guess of guesses.slice(0, n)) {
    if (guess.feedback === "correct" && guess.revealed_color) {
      const group = byColor.get(guess.revealed_color);
      if (group) {
        solved.push(group);
        for (const word of group.words) solvedWords.add(word);
      }
    } else {
      mistakesUsed += 1;
    }
  }

  const remaining = puzzle.board.filter((word) => !solvedWords.has(word));
  return { solved, remaining, mistakesUsed };
}
