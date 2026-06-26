import { describe, it, expect } from "vitest";

import {
  advanceHold,
  appendLetter,
  backspace,
  HOLD_MS,
  INITIAL_HOLD,
  type HoldState,
} from "./word-builder";

describe("advanceHold", () => {
  it("starts a fresh hold when a new confident letter appears", () => {
    const r = advanceHold(INITIAL_HOLD, "A", true, 1000);
    expect(r.state.heldLetter).toBe("A");
    expect(r.state.heldSince).toBe(1000);
    expect(r.locked).toBeNull();
    expect(r.progress).toBe(0);
  });

  it("reports progress while holding, before the threshold", () => {
    const s: HoldState = advanceHold(INITIAL_HOLD, "A", true, 0).state;
    const r = advanceHold(s, "A", true, HOLD_MS / 2);
    expect(r.locked).toBeNull();
    expect(r.progress).toBeCloseTo(0.5, 5);
  });

  it("locks the letter exactly once after the hold threshold", () => {
    const s: HoldState = advanceHold(INITIAL_HOLD, "A", true, 0).state;
    const atThreshold = advanceHold(s, "A", true, HOLD_MS);
    expect(atThreshold.locked).toBe("A");
    expect(atThreshold.progress).toBe(1);
    // Continuing to hold must NOT re-lock (one hold = one letter).
    const stillHeld = advanceHold(atThreshold.state, "A", true, HOLD_MS + 500);
    expect(stillHeld.locked).toBeNull();
  });

  it("requires releasing/changing before the same letter can lock again", () => {
    let s = advanceHold(INITIAL_HOLD, "A", true, 0).state;
    s = advanceHold(s, "A", true, HOLD_MS).state; // locked once
    // Drop to unsure (release), then hold A again → can lock a second A.
    s = advanceHold(s, null, false, HOLD_MS + 100).state;
    s = advanceHold(s, "A", true, HOLD_MS + 200).state; // fresh hold
    const second = advanceHold(s, "A", true, HOLD_MS + 200 + HOLD_MS);
    expect(second.locked).toBe("A");
  });

  it("resets the in-progress hold when the prediction goes unsure", () => {
    const s = advanceHold(INITIAL_HOLD, "A", true, 0).state;
    const r = advanceHold(s, "A", false, 500);
    expect(r.state).toEqual(INITIAL_HOLD);
    expect(r.locked).toBeNull();
  });

  it("restarts the hold timer when the letter changes mid-hold", () => {
    const s = advanceHold(INITIAL_HOLD, "A", true, 0).state;
    const r = advanceHold(s, "B", true, 800);
    expect(r.state.heldLetter).toBe("B");
    expect(r.state.heldSince).toBe(800);
    expect(r.locked).toBeNull();
  });
});

describe("appendLetter / backspace", () => {
  it("appends immutably", () => {
    expect(appendLetter("AB", "C")).toBe("ABC");
  });
  it("backspaces and is safe on empty", () => {
    expect(backspace("ABC")).toBe("AB");
    expect(backspace("")).toBe("");
  });
});
