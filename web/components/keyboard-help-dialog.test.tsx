import { describe, it, expect, vi, beforeEach } from "vitest";
import { render } from "@testing-library/react";

import { KeyboardHelpDialog } from "./keyboard-help-dialog";
import { SHORTCUTS } from "@/lib/keyboard-shortcuts";

// jsdom does not implement the native <dialog> imperative API, so stub it so
// the open/close effect can run without throwing. We assert on the props the
// component passes, not on real top-layer behaviour (that is covered by e2e).
beforeEach(() => {
  HTMLDialogElement.prototype.showModal = vi.fn(function (
    this: HTMLDialogElement,
  ) {
    this.open = true;
  });
  HTMLDialogElement.prototype.close = vi.fn(function (
    this: HTMLDialogElement,
  ) {
    this.open = false;
  });
});

describe("KeyboardHelpDialog", () => {
  it("has an accessible name via aria-labelledby → the h2", () => {
    const { getByRole } = render(
      <KeyboardHelpDialog open={false} onClose={() => {}} />,
    );
    const dialog = getByRole("dialog", { hidden: true });
    expect(dialog).toHaveAttribute("aria-labelledby", "keyboard-help-title");
    const heading = document.getElementById("keyboard-help-title");
    expect(heading?.textContent).toBe("Keyboard shortcuts");
  });

  it("renders a row with a <kbd> for every shortcut", () => {
    const { container } = render(
      <KeyboardHelpDialog open={false} onClose={() => {}} />,
    );
    const kbds = container.querySelectorAll("kbd");
    expect(kbds.length).toBe(SHORTCUTS.length);
    expect([...kbds].map((k) => k.textContent)).toEqual(
      SHORTCUTS.map((s) => s.label),
    );
  });

  it("calls showModal when opened and close when re-closed", () => {
    const { rerender } = render(
      <KeyboardHelpDialog open={false} onClose={() => {}} />,
    );
    expect(HTMLDialogElement.prototype.showModal).not.toHaveBeenCalled();

    rerender(<KeyboardHelpDialog open={true} onClose={() => {}} />);
    expect(HTMLDialogElement.prototype.showModal).toHaveBeenCalledTimes(1);

    rerender(<KeyboardHelpDialog open={false} onClose={() => {}} />);
    expect(HTMLDialogElement.prototype.close).toHaveBeenCalledTimes(1);
  });

  it("invokes onClose from the Close button", () => {
    const onClose = vi.fn();
    const { getByRole } = render(
      <KeyboardHelpDialog open={true} onClose={onClose} />,
    );
    getByRole("button", { name: /close/i }).click();
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
