import type { ReactNode } from "react";

import { Reveal } from "@/components/ui/reveal";
import { cn } from "@/lib/utils";

interface StorySectionProps {
  /** Stable id used as the anchor target and to derive the heading id. */
  id: string;
  /** Visible section heading text (rendered as an h2). */
  title: string;
  /** Optional short kicker/eyebrow shown above the heading. */
  eyebrow?: string;
  className?: string;
  children: ReactNode;
}

/**
 * A consistent narrative section wrapper for the project-story page.
 * Provides a labelled <section> with an h2, keeping heading hierarchy correct
 * (page owns the single h1).
 */
export function StorySection({
  id,
  title,
  eyebrow,
  className,
  children,
}: StorySectionProps) {
  const headingId = `${id}-heading`;
  return (
    <section
      id={id}
      aria-labelledby={headingId}
      className={cn("scroll-mt-24", className)}
    >
      {/* Heading block reveals first, then the content follows on scroll. */}
      <Reveal>
        {eyebrow ? (
          <p className="mb-2 text-sm font-medium uppercase tracking-wide text-accent">
            {eyebrow}
          </p>
        ) : null}
        <h2 id={headingId} className="text-2xl font-bold text-fg sm:text-3xl">
          {title}
        </h2>
      </Reveal>
      <Reveal delay={0.08} className="mt-5 text-pretty leading-relaxed text-fg-muted">
        {children}
      </Reveal>
    </section>
  );
}
