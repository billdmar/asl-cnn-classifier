/**
 * Tiny, SSR-safe theme helpers for the dark/light toggle.
 *
 * The CSS-var foundation lives in globals.css (`:root` = dark default,
 * `[data-theme="light"]` = light) and a no-FOUC inline script in app/layout.tsx
 * sets `data-theme` on <html> from localStorage before first paint. These
 * helpers are the runtime read/write/toggle primitives the client toggle uses;
 * the pure `nextTheme` is trivially unit-testable, and the side-effecting pair
 * is guarded so it's safe to import during SSR.
 */

export type Theme = "dark" | "light";

/** Default theme when nothing is stored or we're not in the browser. */
export const DEFAULT_THEME: Theme = "dark";

const STORAGE_KEY = "theme";

function isTheme(value: string | null): value is Theme {
  return value === "dark" || value === "light";
}

/**
 * Read the persisted theme from localStorage, defaulting to {@link DEFAULT_THEME}.
 * SSR-safe: returns the default when `window` is unavailable. Any malformed
 * stored value also falls back to the default.
 */
export function getStoredTheme(): Theme {
  if (typeof window === "undefined") {
    return DEFAULT_THEME;
  }
  const stored = window.localStorage.getItem(STORAGE_KEY);
  return isTheme(stored) ? stored : DEFAULT_THEME;
}

/**
 * Apply a theme: set `data-theme` on <html> and persist it. SSR-safe no-op when
 * `document` is unavailable.
 */
export function applyTheme(theme: Theme): void {
  if (typeof document === "undefined") {
    return;
  }
  document.documentElement.setAttribute("data-theme", theme);
  window.localStorage.setItem(STORAGE_KEY, theme);
}

/** Pure toggle: dark <-> light. */
export function nextTheme(theme: Theme): Theme {
  return theme === "dark" ? "light" : "dark";
}
