import { useCallback, useEffect, useRef, useState } from "react";
import type { Guess } from "./types";

export interface Resolved {
  index: number; // which guess just resolved
  token: number; // changes every resolution so effects re-fire
}

/**
 * Drives the step-by-step replay.
 *
 * `cursor` counts guesses that have fully resolved; the board shown is the
 * state after `cursor` guesses. When `preview` is true we are mid-animation on
 * `guesses[cursor]`: its four tiles are highlighted and its reasoning shown,
 * but it has not yet been committed into the board.
 *
 * Both Play and the single-step button run the same two-phase cycle —
 * highlight the selection, then resolve it — so stepping shows the selection
 * animation rather than jumping straight to the result. Play keeps going;
 * a step stops once the one guess has resolved.
 */
export function usePlayback(guesses: Guess[], speed: number) {
  const total = guesses.length;
  const [cursor, setCursor] = useState(0);
  const [preview, setPreview] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [stepping, setStepping] = useState(false);
  const [resolved, setResolved] = useState<Resolved | null>(null);
  const timer = useRef<number | null>(null);

  const active = playing || stepping;

  const clear = () => {
    if (timer.current !== null) {
      window.clearTimeout(timer.current);
      timer.current = null;
    }
  };

  // Reset whenever the solve being watched changes (new guess array).
  useEffect(() => {
    clear();
    setCursor(0);
    setPreview(false);
    setPlaying(false);
    setStepping(false);
    setResolved(null);
  }, [guesses]);

  useEffect(() => {
    if (!active) {
      clear();
      return;
    }
    if (cursor >= total) {
      setPlaying(false);
      setStepping(false);
      return;
    }
    const selectMs = 900 / speed;
    const preDelayMs = 240 / speed;
    const settleMs = 520 / speed;
    if (!preview) {
      // Phase 1: highlight the four tiles the agent chose.
      timer.current = window.setTimeout(() => setPreview(true), preDelayMs);
    } else {
      // Phase 2: commit the guess into the board and report the result.
      timer.current = window.setTimeout(() => {
        setPreview(false);
        setResolved({ index: cursor, token: Date.now() });
        setCursor((c) => c + 1);
        if (stepping && !playing) setStepping(false);
      }, selectMs + settleMs);
    }
    return clear;
  }, [active, preview, cursor, total, speed, stepping, playing]);

  const play = useCallback(() => {
    if (cursor >= total) {
      setCursor(0);
      setResolved(null);
    }
    setPreview(false);
    setStepping(false);
    setPlaying(true);
  }, [cursor, total]);

  const pause = useCallback(() => {
    clear();
    setPlaying(false);
    setStepping(false);
    setPreview(false);
  }, []);

  const stepForward = useCallback(() => {
    if (cursor >= total) return;
    clear();
    setPlaying(false);
    setPreview(false);
    setStepping(true); // runs one highlight→resolve cycle, then stops
  }, [cursor, total]);

  const stepBack = useCallback(() => {
    clear();
    setPlaying(false);
    setStepping(false);
    setPreview(false);
    setCursor((c) => Math.max(0, c - 1));
  }, []);

  const seek = useCallback(
    (value: number) => {
      clear();
      setPlaying(false);
      setStepping(false);
      setPreview(false);
      setCursor(Math.max(0, Math.min(total, value)));
    },
    [total],
  );

  const restart = useCallback(() => {
    clear();
    setPlaying(false);
    setStepping(false);
    setPreview(false);
    setCursor(0);
    setResolved(null);
  }, []);

  // The guess currently highlighted (only while previewing before commit).
  const activeGuess = preview && cursor < total ? guesses[cursor] : null;

  return {
    cursor,
    preview,
    playing,
    active,
    activeGuess,
    resolved,
    total,
    play,
    pause,
    stepForward,
    stepBack,
    seek,
    restart,
  };
}
