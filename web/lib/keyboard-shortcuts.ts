/**
 * Pure keyboard-shortcut mapping + matcher. No React, no DOM construction —
 * fully unit-testable via a minimal event shape.
 *
 * Integration contract (see web/lib/use-keyboard-shortcuts.ts): the matcher
 * returns a stable shortcut `id`. Consumers pass a `Record<id, handler>` to
 * `useKeyboardShortcuts`. The handler ids are:
 *   "camera" → start/stop the webcam   (Space)
 *   "copy"   → copy the spelled word   (C)
 *   "reset"  → reset the word          (R)
 *   "share"  → share the result        (S)
 *   "help"   → open this help dialog   (?)
 * The site header wires only "help"; Space/C/R/S are wired into the webcam
 * panel during integration using its existing callbacks + the share helper.
 */

export interface Shortcut {
  /** Stable handler key consumers map to a callback. */
  id: string;
  /** Display key, e.g. "Space", "C", "?". */
  key: string;
  /** Short label for the key column. */
  label: string;
  /** Human description for the help dialog. */
  description: string;
}

export const SHORTCUTS: Shortcut[] = [
  {
    id: "camera",
    key: " ",
    label: "Space",
    description: "Start / stop camera",
  },
  { id: "copy", key: "c", label: "C", description: "Copy spelled word" },
  { id: "reset", key: "r", label: "R", description: "Reset word" },
  { id: "share", key: "s", label: "S", description: "Share result" },
  { id: "help", key: "?", label: "?", description: "Show this help" },
];

/** Map a normalized KeyboardEvent.key to its shortcut id. */
const KEY_TO_ID: Record<string, string> = {
  " ": "camera",
  c: "copy",
  r: "reset",
  s: "share",
  "?": "help",
};

/**
 * True when the event target is a text-editing surface (input/textarea/select
 * or a contenteditable host), so shortcuts must be ignored while typing.
 */
export function isEditableTarget(target: EventTarget | null): boolean {
  if (target === null || !(target instanceof Element)) {
    return false;
  }
  const tag = target.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {
    return true;
  }
  // Treat any contenteditable host as editable. `isContentEditable` is the
  // accurate runtime check in real browsers; we also accept an explicit
  // contentEditable property of "true"/"" (inherited hosts report "inherit"),
  // which covers environments (e.g. jsdom) that don't compute isContentEditable.
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  if (target.isContentEditable === true) {
    return true;
  }
  const editable = target.contentEditable;
  return editable === "true" || editable === "plaintext-only";
}

/** Minimal event shape so the matcher is testable without a real DOM event. */
export interface MatchableKeyEvent {
  key: string;
  target?: EventTarget | null;
  ctrlKey?: boolean;
  metaKey?: boolean;
  altKey?: boolean;
}

/**
 * Returns the matching shortcut id, or null. Null when:
 *  - the target is editable (typing),
 *  - any of ctrl/meta/alt is held (don't hijack browser/OS combos),
 *  - the key maps to no shortcut.
 * Letter keys are matched case-insensitively ("C" or "c" → "copy"). "?" is the
 * Shift+/ glyph and arrives as `key === "?"`.
 */
export function matchShortcut(e: MatchableKeyEvent): string | null {
  if (e.ctrlKey === true || e.metaKey === true || e.altKey === true) {
    return null;
  }
  if (isEditableTarget(e.target ?? null)) {
    return null;
  }
  const normalized = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  return KEY_TO_ID[normalized] ?? null;
}
