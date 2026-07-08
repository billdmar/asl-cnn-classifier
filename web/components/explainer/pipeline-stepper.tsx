"use client";

/**
 * Horizontal step indicator for the inference explainer pipeline.
 *
 * Renders 5 clickable numbered circles connected by lines. Active step gets
 * an accent fill, past steps get a checkmark/filled style, and future steps
 * remain muted. Responsive: labels hide on mobile, circles shrink.
 */

import { motion } from "framer-motion";
import { Check } from "lucide-react";

import { cn } from "@/lib/utils";

export interface PipelineStepperProps {
  activeStep: number; // 1-5
  onStepClick: (step: number) => void;
  labels: string[]; // ["Raw frame", "Hand detection", "Crop", "Tensor channels", "Prediction"]
}

export function PipelineStepper({
  activeStep,
  onStepClick,
  labels,
}: PipelineStepperProps) {
  return (
    <nav
      className="flex w-full items-start justify-between gap-0"
      aria-label="Pipeline steps"
    >
      {labels.map((label, idx) => {
        const step = idx + 1;
        const isPast = step < activeStep;
        const isActive = step === activeStep;
        const isFuture = step > activeStep;

        return (
          <div
            key={step}
            className="flex flex-1 flex-col items-center gap-1.5"
          >
            {/* Circle + connecting line row */}
            <div className="flex w-full items-center">
              {/* Left connector line */}
              {idx > 0 && (
                <div
                  className={cn(
                    "h-0.5 flex-1",
                    isPast || isActive ? "bg-accent" : "bg-border-subtle",
                  )}
                />
              )}

              {/* Step circle */}
              <button
                type="button"
                onClick={() => onStepClick(step)}
                className={cn(
                  "relative flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-xs font-semibold transition-colors md:h-9 md:w-9",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg",
                  isActive &&
                    "bg-accent text-bg shadow-sm shadow-accent/30",
                  isPast &&
                    "bg-accent/20 text-accent border border-accent/40",
                  isFuture &&
                    "bg-surface border border-border-subtle text-fg-muted hover:border-accent/40",
                )}
                aria-current={isActive ? "step" : undefined}
                aria-label={`Step ${step}: ${label}`}
              >
                {/* Active indicator animation */}
                {isActive && (
                  <motion.div
                    layoutId="active-step-indicator"
                    className="absolute inset-0 rounded-full bg-accent"
                    transition={{ type: "spring", stiffness: 400, damping: 30 }}
                    style={{ zIndex: -1 }}
                  />
                )}
                {isPast ? (
                  <Check className="h-3.5 w-3.5" aria-hidden="true" />
                ) : (
                  <span>{step}</span>
                )}
              </button>

              {/* Right connector line */}
              {idx < labels.length - 1 && (
                <div
                  className={cn(
                    "h-0.5 flex-1",
                    isPast ? "bg-accent" : "bg-border-subtle",
                  )}
                />
              )}
            </div>

            {/* Label below the circle (hidden on mobile) */}
            <span
              className={cn(
                "hidden text-center text-xs md:block",
                isActive ? "font-medium text-fg" : "text-fg-muted",
              )}
            >
              {label}
            </span>
          </div>
        );
      })}
    </nav>
  );
}
