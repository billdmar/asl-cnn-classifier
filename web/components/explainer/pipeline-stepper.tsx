"use client";

import { motion } from "framer-motion";

interface PipelineStepperProps {
  activeStep: number;
  onStepClick: (step: number) => void;
  labels: string[];
}

export function PipelineStepper({ activeStep, onStepClick, labels }: PipelineStepperProps) {
  return (
    <div className="flex items-center justify-center gap-0" role="tablist" aria-label="Pipeline steps">
      {labels.map((label, i) => {
        const step = i + 1;
        const isActive = step === activeStep;
        const isPast = step < activeStep;
        return (
          <div key={step} className="flex items-center">
            {i > 0 && (
              <div className={`h-0.5 w-6 sm:w-10 ${isPast ? "bg-accent/60" : "bg-border-subtle"}`} />
            )}
            <button
              role="tab"
              aria-selected={isActive}
              aria-label={`Step ${step}: ${label}`}
              onClick={() => onStepClick(step)}
              className="relative flex flex-col items-center gap-1"
            >
              <div
                className={`flex h-8 w-8 items-center justify-center rounded-full text-xs font-semibold transition-colors ${
                  isActive
                    ? "bg-accent text-white"
                    : isPast
                      ? "bg-accent/20 text-accent"
                      : "border border-border-subtle bg-bg-card text-fg-muted"
                }`}
              >
                {isPast ? "✓" : step}
              </div>
              {isActive && (
                <motion.div
                  layoutId="step-indicator"
                  className="absolute -bottom-1 h-0.5 w-6 rounded-full bg-accent"
                  transition={{ type: "spring", stiffness: 500, damping: 30 }}
                />
              )}
              <span className="hidden text-[10px] text-fg-muted sm:block">{label}</span>
            </button>
          </div>
        );
      })}
    </div>
  );
}
