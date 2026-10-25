import type { Guess } from "../types";

interface Props {
  guesses: Guess[];
  cursor: number;
  onSeek: (value: number) => void;
}

const MARK: Record<string, string> = {
  correct: "●",
  one_away: "◐",
  incorrect: "○",
};

export function GuessLog({ guesses, cursor, onSeek }: Props) {
  return (
    <ol className="guess-log">
      {guesses.map((guess, index) => {
        const resolved = index < cursor;
        const active = index === cursor;
        const cls = [
          "guess-item",
          guess.feedback,
          resolved ? "resolved" : "",
          active ? "active" : "",
        ]
          .filter(Boolean)
          .join(" ");
        return (
          <li key={index} className={cls} onClick={() => onSeek(index)}>
            <span className={`mark ${guess.feedback}`}>{MARK[guess.feedback]}</span>
            <span className="guess-words">{guess.words.join(", ")}</span>
            {guess.fell_back && <span className="tag">fallback</span>}
            {guess.attempts > 1 && <span className="tag">retry×{guess.attempts}</span>}
          </li>
        );
      })}
    </ol>
  );
}
