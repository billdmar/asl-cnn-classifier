/**
 * Unit tests for the SSR-safe theme helpers backing the dark/light toggle.
 *
 * jsdom (the vitest test environment) provides `window`/`document`/
 * `localStorage`, so the side-effecting helpers exercise their real branches
 * here; `nextTheme` is a pure toggle.
 */

import { afterEach, describe, expect, it } from "vitest";

import {
  DEFAULT_THEME,
  applyTheme,
  getStoredTheme,
  nextTheme,
} from "../theme";

afterEach(() => {
  window.localStorage.clear();
  document.documentElement.removeAttribute("data-theme");
});

describe("nextTheme", () => {
  it("toggles dark <-> light", () => {
    expect(nextTheme("dark")).toBe("light");
    expect(nextTheme("light")).toBe("dark");
  });
});

describe("getStoredTheme", () => {
  it("defaults to dark when nothing is stored", () => {
    expect(getStoredTheme()).toBe(DEFAULT_THEME);
    expect(DEFAULT_THEME).toBe("dark");
  });

  it("returns a valid stored theme", () => {
    window.localStorage.setItem("theme", "light");
    expect(getStoredTheme()).toBe("light");
  });

  it("falls back to the default for a malformed stored value", () => {
    window.localStorage.setItem("theme", "neon");
    expect(getStoredTheme()).toBe("dark");
  });
});

describe("applyTheme", () => {
  it("sets the data-theme attribute and persists the choice", () => {
    applyTheme("light");
    expect(document.documentElement.getAttribute("data-theme")).toBe("light");
    expect(window.localStorage.getItem("theme")).toBe("light");

    applyTheme("dark");
    expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
    expect(window.localStorage.getItem("theme")).toBe("dark");
  });

  it("round-trips through getStoredTheme", () => {
    applyTheme("light");
    expect(getStoredTheme()).toBe("light");
  });
});
