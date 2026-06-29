/**
 * Shared motion vocabulary for the site.
 *
 * Single source of truth for the "subtle, fast, premium" taste bar: every
 * animation across the app pulls its timing + variants from here so durations
 * and easing stay consistent (150-400ms, ease-out, transform/opacity only — no
 * layout-affecting properties, so no CLS).
 *
 * These are plain data (framer-motion `Variants`/`Transition` objects), so this
 * module has no "use client" directive and can be imported by both server and
 * client components. Reduced-motion handling lives in the consuming hooks
 * (`components/ui/reveal.tsx`, `lib/use-count-up.ts`), which short-circuit to
 * the final state — and `globals.css` neutralizes timing as a second layer.
 */

import type { Transition, Variants } from "framer-motion";

/** Cubic ease-out — matches the CSS `ease-out` used by the hero's fade-up. */
export const EASE_OUT = [0.22, 1, 0.36, 1] as const;

/** Duration tokens (seconds). Keep everything inside the 150-400ms band. */
export const DUR = { fast: 0.15, base: 0.25, slow: 0.4 } as const;

/** Default transition for reveals/entrances. */
export const transition: Transition = { duration: DUR.base, ease: EASE_OUT };

/** Snappier transition for micro-interactions (hover/tap, small scale-ins). */
export const transitionFast: Transition = { duration: DUR.fast, ease: EASE_OUT };

/**
 * Scroll/entrance reveal: fade in while sliding up a few px. Animates only
 * `opacity` + `transform` (translateY) — never the element's box, so layout is
 * unaffected (zero CLS).
 */
export const revealVariants: Variants = {
  hidden: { opacity: 0, y: 12 },
  visible: { opacity: 1, y: 0, transition },
};

/**
 * Container that staggers its children's reveals. Pair with {@link staggerItem}
 * on each child. `stagger` is the gap (seconds) between successive children.
 */
export function staggerContainer(stagger = 0.06): Variants {
  return {
    hidden: {},
    visible: { transition: { staggerChildren: stagger } },
  };
}

/** Child variant for a {@link staggerContainer}. Same motion as a single reveal. */
export const staggerItem: Variants = {
  hidden: { opacity: 0, y: 12 },
  visible: { opacity: 1, y: 0, transition },
};

/** Scale-in for emphasis (predicted letter, badges). Transform/opacity only. */
export const scaleIn: Variants = {
  hidden: { opacity: 0, scale: 0.92 },
  visible: { opacity: 1, scale: 1, transition: transitionFast },
};

/** `whileHover` props for a subtle lift. Pair with a parent `motion` element. */
export const hoverLift = { y: -2, transition: transitionFast } as const;

/** `whileTap` props for a subtle press. */
export const tapScale = { scale: 0.97 } as const;
