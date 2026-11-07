import { useState } from "react";
import type { RunOptions, StoredSolve } from "../types";

interface Props {
  stored: StoredSolve[];
  activeId: string | null;
  running: boolean;
  runStatus: string | null;
  runError: string | null;
  onSelectStored: (solve: StoredSolve) => void;
  onRun: (options: RunOptions) => void;
}

interface ProviderInfo {
  id: string;
  label: string;
  models: string[];
}

const PROVIDERS: ProviderInfo[] = [
  { id: "deepseek", label: "DeepSeek", models: ["deepseek-chat", "deepseek-reasoner"] },
  {
    id: "anthropic",
    label: "Anthropic (Claude)",
    models: ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"],
  },
];

// deepseek-reasoner does its own chain-of-thought and ignores temperature.
const IGNORES_TEMPERATURE = new Set(["deepseek-reasoner"]);

export function SolvePanel({
  stored,
  activeId,
  running,
  runStatus,
  runError,
  onSelectStored,
  onRun,
}: Props) {
  const [provider, setProvider] = useState("deepseek");
  const [model, setModel] = useState(PROVIDERS[0].models[0]);
  const [hint, setHint] = useState(true);
  const [temperature, setTemperature] = useState(1);
  const current = PROVIDERS.find((p) => p.id === provider)!;
  const tempIgnored = IGNORES_TEMPERATURE.has(model);

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
            const next = PROVIDERS.find((p) => p.id === e.target.value)!;
            setProvider(next.id);
            setModel(next.models[0]);
          }}
        >
          {PROVIDERS.map((p) => (
            <option key={p.id} value={p.id}>{p.label}</option>
          ))}
        </select>
      </label>
      <label className="field">
        <span>Model</span>
        <select value={model} onChange={(e) => setModel(e.target.value)}>
          {current.models.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
      </label>

      <label className="field">
        <span className="field-row">
          Temperature
          <span className="field-value">{tempIgnored ? "—" : temperature.toFixed(1)}</span>
        </span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.1}
          value={temperature}
          disabled={tempIgnored}
          onChange={(e) => setTemperature(Number(e.target.value))}
        />
        {tempIgnored && (
          <span className="field-hint">The reasoner ignores temperature.</span>
        )}
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
            model,
            one_away_hint: hint,
            seed: 7,
            temperature: tempIgnored ? undefined : temperature,
          })
        }
      >
        {running ? runStatus ?? "Running…" : "Run agent"}
      </button>
      {running && (
        <p className="muted small">
          The reasoner can take a minute or two — it’ll replay here when done.
        </p>
      )}
      {runError && <p className="error">{runError}</p>}
      <p className="muted small">
        A live run calls the model's API, costs tokens, and is saved to this
        puzzle's stored solves. It needs the provider's API key in your
        environment or <code>.env</code>.
      </p>
    </div>
  );
}
