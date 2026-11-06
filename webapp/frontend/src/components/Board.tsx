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
  // Order of the four selected words, so tiles pop in one after another.
  const order = new Map<string, number>();
  activeGuess?.words.forEach((word, i) => order.set(word, i));
  const wrong = activeGuess != null && activeGuess.feedback !== "correct";
  const correct = activeGuess != null && activeGuess.feedback === "correct";

  return (
    <div className="board">
      {state.solved.map((group) => (
        <div key={group.color} className={`solved-row ${group.color}`}>
          <div className="solved-title">
            {COLOR_LABEL[group.color]}: {group.title}
          </div>
          <div className="solved-words">{group.words.join(" · ")}</div>
        </div>
      ))}

      {state.remaining.length > 0 && (
        <div className="tile-grid">
          {state.remaining.map((word) => {
            const picked = order.has(word);
            const cls = [
              "tile",
              picked ? "tile-selected" : "",
              picked && correct ? "tile-correct" : "",
              picked && wrong ? "tile-wrong" : "",
            ]
              .filter(Boolean)
              .join(" ");
            const style = picked
              ? { animationDelay: `${(order.get(word) ?? 0) * 70}ms` }
              : undefined;
            return (
              <div key={word} className={cls} style={style}>
                <span>{word}</span>
              </div>
            );
          })}
        </div>
      )}

      <div className="mistakes">
        <span className="mistakes-label">Mistakes remaining</span>
        <span className="dots">
          {[0, 1, 2, 3].map((i) => (
            <span
              key={i}
              className={`mdot ${i < 4 - state.mistakesUsed ? "on" : "off"}`}
            />
          ))}
        </span>
      </div>
    </div>
  );
}
