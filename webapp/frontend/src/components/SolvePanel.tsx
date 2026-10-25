import { useState } from "react";
import type { RunOptions, StoredSolve } from "../types";

interface Props {
  stored: StoredSolve[];
  activeId: string | null;
  running: boolean;
  runError: string | null;
  onSelectStored: (solve: StoredSolve) => void;
  onRun: (options: RunOptions) => void;
}

const PROVIDERS = [
  { id: "anthropic", label: "Anthropic (Claude)", defaultModel: "claude-opus-5-5" },
  { id: "deepseek", label: "DeepSeek", defaultModel: "deepseek-chat" },
];

export function SolvePanel({
  stored,
  activeId,
  running,
  runError,
  onSelectStored,
  onRun,
}: Props) {
  const [provider, setProvider] = useState("anthropic");
  const [model, setModel] = useState("");
  const [hint, setHint] = useState(true);
  const current = PROVIDERS.find((p) => p.id === provider)!;

  return (
    <div className="solve-panel">
      <h3>Stored solves</h3>
      {stored.length === 0 ? (
        <p className="muted">No saved run for this puzzle. Run the agent below.</p>
      ) : (
        <ul className="solve-list">
          {stored.map((solve) => (
            <li
              key={solve.id}
              className={solve.id === activeId ? "active" : ""}
              onClick={() => onSelectStored(solve)}
            >
              <span className={`dot ${solve.status}`} />
              <span className="solve-label">{solve.label}</span>
              <span className="solve-meta">
                {solve.status} · {solve.groups_solved}/4 · {solve.turns} turns
              </span>
            </li>
          ))}
        </ul>
      )}

      <h3>Run the agent live</h3>
      <label className="field">
        <span>Provider</span>
        <select
          value={provider}
          onChange={(e) => {
            setProvider(e.target.value);
            setModel("");
          }}
        >
          {PROVIDERS.map((p) => (
            <option key={p.id} value={p.id}>{p.label}</option>
          ))}
        </select>
      </label>
      <label className="field">
        <span>Model</span>
        <input
          type="text"
          value={model}
          placeholder={current.defaultModel}
          onChange={(e) => setModel(e.target.value)}
        />
      </label>
      <label className="checkbox">
        <input
          type="checkbox"
          checked={hint}
          onChange={(e) => setHint(e.target.checked)}
        />
        <span>One-away recovery hint</span>
      </label>

      <button
        className="run-button"
        disabled={running}
        onClick={() =>
          onRun({
            provider,
            model: model.trim() || undefined,
            one_away_hint: hint,
            seed: 7,
          })
        }
      >
        {running ? "Running… (watch it live)" : "Run agent"}
      </button>
      {runError && <p className="error">{runError}</p>}
      <p className="muted small">
        A live run calls the model's API and costs tokens. It needs the
        provider's API key in your environment or <code>.env</code>.
      </p>
    </div>
  );
}
