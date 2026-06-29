import { describe, it, expect } from "vitest";

import {
  isEditableTarget,
  matchShortcut,
  SHORTCUTS,
} from "@/lib/keyboard-shortcuts";

describe("SHORTCUTS table", () => {
  it("has the expected ids and KeyboardEvent.key values", () => {
    expect(SHORTCUTS.map((s) => [s.id, s.key])).toEqual([
      ["camera", " "],
      ["copy", "c"],
      ["reset", "r"],
      ["share", "s"],
      ["help", "?"],
    ]);
  });
});

describe("matchShortcut", () => {
  it("maps each key to its shortcut id", () => {
    expect(matchShortcut({ key: " " })).toBe("camera");
    expect(matchShortcut({ key: "c" })).toBe("copy");
    expect(matchShortcut({ key: "r" })).toBe("reset");
    expect(matchShortcut({ key: "s" })).toBe("share");
    expect(matchShortcut({ key: "?" })).toBe("help");
  });

  it("is case-insensitive for letter keys", () => {
    expect(matchShortcut({ key: "C" })).toBe("copy");
    expect(matchShortcut({ key: "R" })).toBe("reset");
    expect(matchShortcut({ key: "S" })).toBe("share");
  });

  it("returns null for unmapped keys", () => {
    expect(matchShortcut({ key: "x" })).toBeNull();
    expect(matchShortcut({ key: "Tab" })).toBeNull();
    expect(matchShortcut({ key: "Enter" })).toBeNull();
  });

  it("returns null when a modifier key is held", () => {
    expect(matchShortcut({ key: "c", ctrlKey: true })).toBeNull();
    expect(matchShortcut({ key: "c", metaKey: true })).toBeNull();
    expect(matchShortcut({ key: "c", altKey: true })).toBeNull();
    expect(matchShortcut({ key: " ", metaKey: true })).toBeNull();
  });

  it("returns null when typing in an editable target", () => {
    for (const tag of ["input", "textarea", "select"] as const) {
      const el = document.createElement(tag);
      expect(matchShortcut({ key: "c", target: el })).toBeNull();
    }
    const editable = document.createElement("div");
    editable.contentEditable = "true";
    expect(matchShortcut({ key: "c", target: editable })).toBeNull();
  });

  it("still matches when the target is a non-editable element", () => {
    const button = document.createElement("button");
    expect(matchShortcut({ key: "c", target: button })).toBe("copy");
  });
});

describe("isEditableTarget", () => {
  it("is false for null and non-element targets", () => {
    expect(isEditableTarget(null)).toBe(false);
  });

  it("is true for inputs, textareas, selects and contenteditable", () => {
    expect(isEditableTarget(document.createElement("input"))).toBe(true);
    expect(isEditableTarget(document.createElement("textarea"))).toBe(true);
    expect(isEditableTarget(document.createElement("select"))).toBe(true);

    const editable = document.createElement("div");
    editable.contentEditable = "true";
    expect(isEditableTarget(editable)).toBe(true);
  });

  it("is false for a plain element", () => {
    expect(isEditableTarget(document.createElement("div"))).toBe(false);
  });
});
