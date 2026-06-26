/**
 * Fingerspelling word-builder: turn the live single-letter classifier into a
 * "spell a word by holding letters" interaction. Hold a confident, stable letter
 * for {@link HOLD_MS} and it locks into a running word; you must then drop or
 * change the letter before the same one can lock again (one hold = one letter),
 * so holding "A" appends a single A, not a stream.
 *
 * The logic here is pure and frame-driven (no timers, no DOM) so it's unit-tested
 * in isolation and stays SSR-safe. The webcam panel calls {@link advanceHold} on
 * each classify tick with the already-smoothed top letter.
 */

/** How long (ms) a letter must be held steady before it locks into the word. */
export const HOLD_MS = 1500;

/** In-progress hold state. `lockedKey` debounces repeat-locks of one hold. */
export interface HoldState {
  /** The letter currently being held, or null when none/unsure/hand-lost. */
  heldLetter: string | null;
  /** Timestamp (ms) the current hold began. */
  heldSince: number;
  /**
   * The letter+startTime that has already locked, so a continued hold doesn't
   * re-lock. Cleared when the held letter changes or resets.
   */
  lockedThisHold: boolean;
}

export const INITIAL_HOLD: HoldState = {
  heldLetter: null,
  heldSince: 0,
  lockedThisHold: false,
};

export interface HoldResult {
  /** The next hold state to store. */
  state: HoldState;
  /** A letter to append to the word this tick, or null. */
  locked: string | null;
  /** 0–1 progress of the current hold, for a visual ring/bar. */
  progress: number;
}

/**
 * Advance the hold state by one classify tick. Pure.
 *
 * @param state - Previous hold state.
 * @param letter - The current smoothed top-1 letter (ignored when not confident).
 * @param isConfident - False when the prediction is "unsure" or no hand is shown.
 * @param now - Current timestamp in ms.
 * @param holdMs - Hold threshold (defaults to {@link HOLD_MS}).
 */
export function advanceHold(
  state: HoldState,
  letter: string | null,
  isConfident: boolean,
  now: number,
  holdMs: number = HOLD_MS,
): HoldResult {
  // No confident letter (unsure / hand lost): reset the in-progress hold so a
  // fresh, deliberate hold is required next time. The built word is untouched.
  if (!isConfident || !letter) {
    return { state: INITIAL_HOLD, locked: null, progress: 0 };
  }

  // A new/changed letter starts a fresh hold.
  if (letter !== state.heldLetter) {
    return {
      state: { heldLetter: letter, heldSince: now, lockedThisHold: false },
      locked: null,
      progress: 0,
    };
  }

  const elapsed = now - state.heldSince;
  const progress = Math.max(0, Math.min(1, elapsed / holdMs));

  // Threshold reached and this hold hasn't locked yet → emit one lock.
  if (elapsed >= holdMs && !state.lockedThisHold) {
    return {
      state: { ...state, lockedThisHold: true },
      locked: letter,
      progress: 1,
    };
  }

  // Same letter, still holding (either pre-threshold, or already locked).
  return { state, locked: null, progress };
}

/** Append a locked letter to the word. Pure (returns a new string). */
export function appendLetter(word: string, letter: string): string {
  return word + letter;
}

/** Remove the last letter (backspace). Pure. */
export function backspace(word: string): string {
  return word.slice(0, -1);
}
