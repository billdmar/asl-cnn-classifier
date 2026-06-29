import { cn } from "@/lib/utils";

/**
 * Dark-themed loading placeholder for lazy-loaded sections. Reserves a fixed
 * height so swapping in the real (code-split) component causes no layout shift
 * (CLS). Announced politely to assistive tech via role="status" + aria-busy.
 *
 * A premium gradient shimmer (S0 `animate-shimmer`, a background-position sweep
 * — transform/opacity-free, so no reflow) replaces a plain pulse. Under
 * `prefers-reduced-motion`, globals.css neutralizes the shimmer keyframe.
 */
interface SkeletonCardProps {
  /** Min height to reserve, matching the eventual component's footprint. */
  minHeight: number;
  /** Accessible label, e.g. "Loading live demo". */
  label: string;
}

export function SkeletonCard({ minHeight, label }: SkeletonCardProps) {
  return (
    <div
      role="status"
      aria-busy="true"
      aria-label={label}
      style={{ minHeight }}
      className="relative flex items-center justify-center overflow-hidden rounded-xl border border-border bg-bg-card shadow-sm"
    >
      {/* Shimmer sweep: a translucent highlight panned across via background-
          position (compositor-only, no reflow). Disabled under reduced motion
          by globals.css; the static gradient remains, so there's no flash. */}
      <span
        aria-hidden="true"
        className={cn(
          "pointer-events-none absolute inset-0",
          "bg-gradient-to-r from-transparent via-fg/10 to-transparent",
          "bg-[length:200%_100%] animate-shimmer",
        )}
      />
      <span className="relative text-sm text-fg-muted">{label}…</span>
    </div>
  );
}
