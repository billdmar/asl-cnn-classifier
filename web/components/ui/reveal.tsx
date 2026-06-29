"use client";

/**
 * Scroll-reveal wrapper — the workhorse entrance animation consumed across the
 * landing page and the About story.
 *
 * Contract that keeps the Lighthouse + axe gates green:
 * - Animates ONLY opacity + translateY (12px). Never animates the element's
 *   box (width/height/margin), so it contributes zero CLS.
 * - Does NOT use AnimatePresence and never unmounts children, so focus order
 *   and ARIA landmarks are untouched.
 * - Under `prefers-reduced-motion` it renders the final visible state with no
 *   initial hidden state and no transition (the content is never stuck at
 *   opacity:0). `globals.css` neutralizes timing as a second, independent layer.
 * - Never wrap an LCP element (hero/about <h1>) in this — those keep the CSS
 *   `rise-up`/`fade-up` so they paint immediately.
 */

import { motion, useInView, useReducedMotion } from "framer-motion";
import { type ElementType, type ReactNode, useRef } from "react";

import { cn } from "@/lib/utils";
import { revealVariants, staggerContainer, staggerItem } from "@/lib/motion";

export interface RevealProps {
  children: ReactNode;
  /** Element to render (default "div"). */
  as?: ElementType;
  /** Extra classes, merged via `cn`. */
  className?: string;
  /** Delay (seconds) before this reveal starts — for manual sequencing. */
  delay?: number;
  /** Stagger direct children instead of revealing as one block. */
  stagger?: boolean;
  /** Gap (seconds) between staggered children. Only used when `stagger`. */
  staggerGap?: number;
  /**
   * IntersectionObserver root margin (CSS-margin syntax, px or %). Default fires
   * slightly before the element is fully in view.
   */
  margin?: string;
  /** Animate only the first time it enters (default true). */
  once?: boolean;
}

/**
 * Reveal a block (or, with `stagger`, its direct children one after another) as
 * it scrolls into view.
 */
export function Reveal({
  children,
  as = "div",
  className,
  delay = 0,
  stagger = false,
  staggerGap = 0.06,
  margin = "0px 0px -10% 0px",
  once = true,
}: RevealProps) {
  const reduce = useReducedMotion();
  const ref = useRef<HTMLElement>(null);
  // useInView's `margin` is typed loosely upstream; our prop constrains the shape.
  const inView = useInView(ref, { once, margin: margin as never });

  const MotionTag = motion(as as ElementType);

  // Reduced motion: render visible immediately, no hidden state, no transition.
  if (reduce) {
    const StaticTag = as;
    return <StaticTag className={className}>{children}</StaticTag>;
  }

  const variants = stagger ? staggerContainer(staggerGap) : revealVariants;

  return (
    <MotionTag
      ref={ref}
      className={cn(className)}
      variants={variants}
      initial="hidden"
      animate={inView ? "visible" : "hidden"}
      transition={delay ? { delay } : undefined}
    >
      {children}
    </MotionTag>
  );
}

export interface RevealItemProps {
  children: ReactNode;
  as?: ElementType;
  className?: string;
}

/**
 * A single child inside a `<Reveal stagger>`. Reveals in sequence with its
 * siblings. Under reduced motion it renders plainly (the parent already
 * short-circuits, so this is only reached when motion is enabled).
 */
export function RevealItem({ children, as = "div", className }: RevealItemProps) {
  const MotionTag = motion(as as ElementType);
  return (
    <MotionTag className={cn(className)} variants={staggerItem}>
      {children}
    </MotionTag>
  );
}
