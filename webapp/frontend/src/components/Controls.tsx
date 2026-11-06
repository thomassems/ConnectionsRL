interface Props {
  cursor: number;
  total: number;
  playing: boolean;
  onPlay: () => void;
  onPause: () => void;
  onStepBack: () => void;
  onStepForward: () => void;
  onRestart: () => void;
  onSeek: (value: number) => void;
}

export function Controls({
  cursor,
  total,
  playing,
  onPlay,
  onPause,
  onStepBack,
  onStepForward,
  onRestart,
  onSeek,
}: Props) {
  return (
    <div className="controls">
      <div className="status-line">
        {cursor >= total ? "Replay finished" : `Guess ${cursor + 1} of ${total}`}
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
    </div>
  );
}
