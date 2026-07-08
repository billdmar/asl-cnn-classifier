"use client";

/**
 * Interactive temperature slider for the prediction step.
 *
 * Re-applies temperature scaling to the raw logits on every change and passes
 * the new probability distribution to the parent.
 */

import { useCallback, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { applyTemperature } from "@/lib/inference";

export interface TemperatureSliderProps {
  logits: Float32Array;
  onChange: (probs: Float32Array, temperature: number) => void;
}

export function TemperatureSlider({ logits, onChange }: TemperatureSliderProps) {
  const [temperature, setTemperature] = useState(1.0);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const t = parseFloat(e.target.value);
      setTemperature(t);
      const probs = applyTemperature(logits, t);
      onChange(probs, t);
    },
    [logits, onChange],
  );

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-3">
        <label
          htmlFor="temperature-slider"
          className="text-sm font-medium text-fg"
        >
          Temperature
        </label>
        <Badge variant="accent" className="font-mono">
          T = {temperature.toFixed(2)}
        </Badge>
      </div>
      <input
        id="temperature-slider"
        type="range"
        min="0.5"
        max="3.0"
        step="0.05"
        value={temperature}
        onChange={handleChange}
        className="h-2 w-full cursor-pointer appearance-none rounded-full bg-border accent-accent [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-accent"
        aria-label="Temperature scaling factor"
      />
      <div className="flex justify-between text-[11px] text-fg-muted">
        <span>Sharper (more confident)</span>
        <span>Calibrated</span>
        <span>Softer (less confident)</span>
      </div>
    </div>
  );
}
