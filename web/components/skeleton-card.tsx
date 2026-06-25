/**
 * Dark-themed loading placeholder for lazy-loaded sections. Reserves a fixed
 * height so swapping in the real (code-split) component causes no layout shift
 * (CLS). Announced politely to assistive tech via role="status" + aria-busy.
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
      className="flex animate-pulse items-center justify-center rounded-xl border border-border bg-bg-card shadow-sm"
    >
      <span className="text-sm text-fg-muted">{label}…</span>
    </div>
  );
}
