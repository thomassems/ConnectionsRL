import type { Guess } from "../types";

interface Props {
  cursor: number;
  total: number;
  playing: boolean;
  activeGuess: Guess | null;
  speed: number;
  onPlay: () => void;
  onPause: () => void;
  onStepBack: () => void;
  onStepForward: () => void;
  onRestart: () => void;
  onSeek: (value: number) => void;
  onSpeed: (value: number) => void;
}

const FEEDBACK_TEXT: Record<string, string> = {
  correct: "Correct",
  one_away: "One away…",
  incorrect: "Incorrect",
};

export function Controls({
  cursor,
  total,
  playing,
  activeGuess,
  speed,
  onPlay,
  onPause,
  onStepBack,
  onStepForward,
  onRestart,
  onSeek,
  onSpeed,
}: Props) {
  const banner = activeGuess
    ? { text: FEEDBACK_TEXT[activeGuess.feedback], kind: activeGuess.feedback }
    : null;

  return (
    <div className="controls">
      <div className={`feedback-banner ${banner ? banner.kind : "empty"}`}>
        {banner ? banner.text : `Guess ${Math.min(cursor + 1, total)} of ${total}`}
      </div>

      <input
        className="scrubber"
        type="range"
        min={0}
        max={total}
        value={cursor}
        onChange={(e) => onSeek(Number(e.target.value))}
      />

      <div className="control-row">
        <button onClick={onRestart} title="Restart">⟲</button>
        <button onClick={onStepBack} disabled={cursor === 0} title="Step back">◀</button>
        {playing ? (
          <button className="primary" onClick={onPause}>Pause</button>
        ) : (
          <button className="primary" onClick={onPlay} disabled={total === 0}>
            {cursor >= total ? "Replay" : "Play"}
          </button>
        )}
        <button onClick={onStepForward} disabled={cursor >= total} title="Step forward">▶</button>
      </div>

      <div className="speed-row">
        <span>Speed</span>
        {[0.5, 1, 2, 4].map((value) => (
          <button
            key={value}
            className={speed === value ? "chip active" : "chip"}
            onClick={() => onSpeed(value)}
          >
            {value}×
          </button>
        ))}
      </div>
    </div>
  );
}
