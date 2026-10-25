import { useCallback, useEffect, useRef, useState } from "react";
import type { Guess } from "./types";

/**
 * Drives the step-by-step replay.
 *
 * `cursor` counts guesses that have fully resolved; the board shown is the
 * state after `cursor` guesses. When `preview` is true we are mid-animation on
 * `guesses[cursor]`: its four tiles are highlighted and its feedback shown, but
 * it has not yet been committed into the board. Playing alternates
 * preview (highlight) → commit (cursor + 1) with pauses between.
 */
export function usePlayback(guesses: Guess[], speed: number) {
  const total = guesses.length;
  const [cursor, setCursor] = useState(0);
  const [preview, setPreview] = useState(false);
  const [playing, setPlaying] = useState(false);
  const timer = useRef<number | null>(null);

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
  }, [guesses]);

  useEffect(() => {
    if (!playing) {
      clear();
      return;
    }
    if (cursor >= total) {
      setPlaying(false);
      return;
    }
    const selectMs = 850 / speed;
    const commitGapMs = 650 / speed;
    if (!preview) {
      timer.current = window.setTimeout(() => setPreview(true), 250 / speed);
    } else {
      timer.current = window.setTimeout(() => {
        setPreview(false);
        setCursor((c) => c + 1);
      }, selectMs + commitGapMs);
    }
    return clear;
  }, [playing, preview, cursor, total, speed]);

  const play = useCallback(() => {
    if (cursor >= total) {
      setCursor(0);
      setPreview(false);
    }
    setPlaying(true);
  }, [cursor, total]);

  const pause = useCallback(() => {
    setPlaying(false);
    setPreview(false);
  }, []);

  const stepForward = useCallback(() => {
    clear();
    setPlaying(false);
    setPreview(false);
    setCursor((c) => Math.min(total, c + 1));
  }, [total]);

  const stepBack = useCallback(() => {
    clear();
    setPlaying(false);
    setPreview(false);
    setCursor((c) => Math.max(0, c - 1));
  }, []);

  const seek = useCallback(
    (value: number) => {
      clear();
      setPlaying(false);
      setPreview(false);
      setCursor(Math.max(0, Math.min(total, value)));
    },
    [total],
  );

  const restart = useCallback(() => {
    clear();
    setPlaying(false);
    setPreview(false);
    setCursor(0);
  }, []);

  // The guess currently highlighted (only while previewing before commit).
  const activeGuess = preview && cursor < total ? guesses[cursor] : null;

  return {
    cursor,
    preview,
    playing,
    activeGuess,
    total,
    play,
    pause,
    stepForward,
    stepBack,
    seek,
    restart,
  };
}
