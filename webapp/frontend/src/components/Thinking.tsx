import type { Guess } from "../types";

interface Props {
  guess: Guess | null;
}

/**
 * Shows what the agent was thinking while a guess is highlighted. Live runs
 * capture the model's reasoning; older stored reports have none, so we fall
 * back to a quiet note rather than inventing a rationale.
 */
export function Thinking({ guess }: Props) {
  if (!guess) return null;
  return (
    <div className="thinking">
      <div className="thinking-head">
        <span className="thinking-dot" />
        Agent’s reasoning
      </div>
      {guess.rationale ? (
        <p className="thinking-body">{guess.rationale}</p>
      ) : (
        <p className="thinking-body muted">
          No saved reasoning for this guess — run the agent live to see it think.
        </p>
      )}
    </div>
  );
}
