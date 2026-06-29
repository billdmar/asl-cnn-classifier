"use client";

import { useEffect } from "react";

import { matchShortcut } from "@/lib/keyboard-shortcuts";

/**
 * Attaches a single window "keydown" listener and dispatches to the handler
 * mapped from the shortcut id (see SHORTCUTS in lib/keyboard-shortcuts.ts).
 *
 * - SSR-safe: the listener is attached inside an effect, so `window` is only
 *   touched in the browser; it is cleaned up on unmount.
 * - Only calls preventDefault for keys that actually map to a provided handler
 *   (notably Space, which would otherwise scroll the page). Tab and every
 *   unmapped key pass through untouched, so focus order is never hijacked.
 * - matchShortcut already returns null while typing in an editable field, so
 *   we never swallow or preventDefault keystrokes there.
 *
 * @param handlers map of shortcut id → callback (e.g. `{ help: () => ... }`).
 */
export function useKeyboardShortcuts(
  handlers: Record<string, () => void>,
): void {
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const id = matchShortcut(e);
      if (id === null) {
        return;
      }
      const handler = handlers[id];
      if (handler === undefined) {
        return;
      }
      e.preventDefault();
      handler();
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [handlers]);
}
