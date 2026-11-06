interface Props {
  message: string | null;
  kind: string;
}

/** NYT-style pill that pops in over the board (e.g. "One away…"). */
export function Toast({ message, kind }: Props) {
  if (!message) return null;
  return (
    <div className="toast-layer">
      <div className={`toast ${kind}`}>{message}</div>
    </div>
  );
}
