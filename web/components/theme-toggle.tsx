"use client";

import { useEffect, useState } from "react";
import { Moon, Sun } from "lucide-react";

import { applyTheme, getStoredTheme, nextTheme, type Theme } from "@/lib/theme";

/**
 * Dark/light theme toggle.
 *
 * The no-FOUC inline script in app/layout.tsx already set `data-theme` on
 * <html> before first paint, so on mount we adopt that *live* attribute rather
 * than re-reading localStorage (which could disagree if it were ever out of
 * sync). The rendered icon depends on this client-only state, so we render a
 * neutral, non-icon placeholder until `mounted` flips true — that keeps the
 * server-rendered markup and the first client render identical and avoids a
 * hydration mismatch.
 */
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("dark");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const attr = document.documentElement.getAttribute("data-theme");
    setTheme(attr === "light" || attr === "dark" ? attr : getStoredTheme());
    setMounted(true);
  }, []);

  const handleClick = () => {
    const next = nextTheme(theme);
    applyTheme(next);
    setTheme(next);
  };

  // In dark mode the button switches *to* light, so it shows a Sun; in light
  // mode it shows a Moon. The label describes the action, not the current state.
  const goingToLight = theme === "dark";
  const label = goingToLight ? "Switch to light theme" : "Switch to dark theme";

  return (
    <button
      type="button"
      onClick={handleClick}
      aria-label={label}
      aria-pressed={theme === "light"}
      className="flex h-9 w-9 items-center justify-center rounded-md text-fg-muted transition-colors hover:bg-bg-card hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg"
    >
      {!mounted ? (
        // Stable placeholder until client state is known — matches icon size so
        // layout doesn't shift, and renders identically on server + first paint.
        <span className="h-5 w-5" aria-hidden="true" />
      ) : goingToLight ? (
        <Sun className="h-5 w-5" aria-hidden="true" />
      ) : (
        <Moon className="h-5 w-5" aria-hidden="true" />
      )}
    </button>
  );
}
