"use client";

import { useCallback, useState } from "react";
import { applyTemperature } from "@/lib/inference";
import { Badge } from "@/components/ui/badge";

interface TemperatureSliderProps {
  logits: Float32Array;
  onChange: (probs: Float32Array, temperature: number) => void;
}

export function TemperatureSlider({ logits, onChange }: TemperatureSliderProps) {
  const [temperature, setTemperature] = useState(1.0);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const t = parseFloat(e.target.value);
      setTemperature(t);
      onChange(applyTemperature(logits, t), t);
    },
    [logits, onChange],
  );

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm text-fg-muted">Temperature scaling</span>
        <Badge variant="accent">T = {temperature.toFixed(2)}</Badge>
      </div>
      <input
        type="range"
        min="0.5"
        max="3.0"
        step="0.05"
        value={temperature}
        onChange={handleChange}
        className="w-full accent-[var(--color-accent,#7c5cff)]"
        aria-label="Temperature scaling factor"
      />
      <div className="flex justify-between text-xs text-fg-muted">
        <span>Sharper</span>
        <span>Calibrated</span>
        <span>Softer</span>
      </div>
    </div>
  );
}
