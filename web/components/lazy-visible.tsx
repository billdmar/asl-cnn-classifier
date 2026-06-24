"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";

/**
 * Defers mounting heavy, below-the-fold content until the user is about to see
 * it — keeping its code-split JS (onnxruntime-web, recharts, @mediapipe) and
 * assets (example images) off the critical path so they don't compete with the
 * hero's Largest Contentful Paint.
 *
 * Activation is triggered by EITHER:
 *   - the placeholder scrolling near the viewport (IntersectionObserver), or
 *   - an idle/timeout fallback after first paint.
 *
 * The fallback matters for two reasons: it guarantees the content eventually
 * mounts even if the user never scrolls (and for non-scrolling automation /
 * E2E), and it degrades gracefully where IntersectionObserver is unavailable.
 * SSR-safe: all browser APIs are touched only inside the effect, and the server
 * renders the placeholder.
 */
interface LazyVisibleProps {
  children: ReactNode;
  /** Accessible placeholder shown until the content mounts. */
  placeholder: ReactNode;
  /** Idle fallback delay (ms) after which content mounts regardless of scroll. */
  fallbackDelayMs?: number;
}

export function LazyVisible({
  children,
  placeholder,
  fallbackDelayMs = 2500,
}: LazyVisibleProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [show, setShow] = useState(false);

  useEffect(() => {
    if (show) return;

    const activate = () => setShow(true);

    // Idle/timeout fallback: ensures the section mounts even without scrolling.
    const timer = window.setTimeout(activate, fallbackDelayMs);

    const node = ref.current;
    if (node && typeof IntersectionObserver !== "undefined") {
      const observer = new IntersectionObserver(
        (entries) => {
          if (entries.some((e) => e.isIntersecting)) {
            window.clearTimeout(timer);
            observer.disconnect();
            activate();
          }
        },
        // Start loading a little before the section enters the viewport.
        { rootMargin: "200px 0px" },
      );
      observer.observe(node);
      return () => {
        window.clearTimeout(timer);
        observer.disconnect();
      };
    }

    return () => window.clearTimeout(timer);
  }, [show, fallbackDelayMs]);

  return <div ref={ref}>{show ? children : placeholder}</div>;
}
