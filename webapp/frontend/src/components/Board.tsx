import type { Guess } from "../types";
import type { ReplayState } from "../replay";

interface Props {
  state: ReplayState;
  activeGuess: Guess | null;
}

const COLOR_LABEL: Record<string, string> = {
  yellow: "Yellow",
  green: "Green",
  blue: "Blue",
  purple: "Purple",
};

export function Board({ state, activeGuess }: Props) {
  const selected = new Set(activeGuess?.words ?? []);
  // Shake the highlighted tiles only on a wrong guess.
  const wrong = activeGuess != null && activeGuess.feedback !== "correct";
  const correct = activeGuess != null && activeGuess.feedback === "correct";

  return (
    <div className="board">
      {state.solved.map((group) => (
        <div key={group.color} className={`solved-row ${group.color}`}>
          <div className="solved-title">{COLOR_LABEL[group.color]}: {group.title}</div>
          <div className="solved-words">{group.words.join(" · ")}</div>
        </div>
      ))}

      {state.remaining.length > 0 && (
        <div className="tile-grid">
          {state.remaining.map((word) => {
            const isSelected = selected.has(word);
            const cls = [
              "tile",
              isSelected && correct ? "tile-correct" : "",
              isSelected && wrong ? "tile-wrong" : "",
              isSelected ? "tile-selected" : "",
            ]
              .filter(Boolean)
              .join(" ");
            return (
              <div key={word} className={cls}>
                <span>{word}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
