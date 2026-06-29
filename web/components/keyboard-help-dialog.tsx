"use client";

import { useEffect, useRef } from "react";

import { SHORTCUTS } from "@/lib/keyboard-shortcuts";

interface KeyboardHelpDialogProps {
  open: boolean;
  onClose: () => void;
}

/**
 * Keyboard shortcuts help, rendered as a native <dialog> driven by
 * showModal()/close(). The native element gives us a free focus trap, Escape
 * handling, top-layer stacking and a backdrop. It is axe-clean because it has
 * an accessible name via aria-labelledby → the internal <h2>.
 *
 * Closed by default, so it adds nothing to the initial tab order or DOM that
 * axe would flag on the landing render.
 */
export function KeyboardHelpDialog({ open, onClose }: KeyboardHelpDialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);

  // Sync the imperative <dialog> state with the `open` prop. Guarded for SSR
  // (effects only run in the browser) and a null ref.
  useEffect(() => {
    const dialog = dialogRef.current;
    if (dialog === null) {
      return;
    }
    if (open && !dialog.open) {
      dialog.showModal();
    } else if (!open && dialog.open) {
      dialog.close();
    }
  }, [open]);

  return (
    <dialog
      ref={dialogRef}
      aria-labelledby="keyboard-help-title"
      onClose={onClose}
      className="kbd-help-dialog m-auto w-[min(28rem,calc(100vw-2rem))] rounded-xl border border-border bg-bg-card p-0 text-fg shadow-2xl backdrop:bg-black/60"
    >
      <div className="flex items-center justify-between border-b border-border px-5 py-4">
        <h2 id="keyboard-help-title" className="text-lg font-semibold text-fg">
          Keyboard shortcuts
        </h2>
        <button
          type="button"
          onClick={onClose}
          className="rounded-md px-2 py-1 text-sm text-fg-muted transition-colors hover:bg-bg hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg-card"
        >
          Close
        </button>
      </div>

      <ul className="flex flex-col gap-1 px-5 py-4">
        {SHORTCUTS.map((shortcut) => (
          <li
            key={shortcut.id}
            className="flex items-center justify-between gap-4 py-1.5"
          >
            <span className="text-sm text-fg-muted">
              {shortcut.description}
            </span>
            <kbd className="inline-flex min-w-8 items-center justify-center rounded-md border border-border bg-bg px-2 py-1 font-mono text-xs text-fg">
              {shortcut.label}
            </kbd>
          </li>
        ))}
      </ul>
    </dialog>
  );
}
