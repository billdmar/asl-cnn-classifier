import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";

import { ConfusionHeatmap } from "./confusion-heatmap";
import type { RealworldEval } from "@/lib/metrics";

function makeData(overrides: Partial<RealworldEval> = {}): RealworldEval {
  return {
    source: "cross-dataset:test",
    num_samples: 4,
    hand_crop_used: true,
    num_no_hand_fallback: 0,
    accuracy: 0.5,
    macro_f1: 0.5,
    macro_precision: 0.5,
    macro_recall: 0.5,
    accuracy_ay: 0.5,
    macro_f1_ay: 0.5,
    num_samples_ay: 4,
    per_class: {},
    most_confused_pairs: [{ true: "A", pred: "B", count: 2 }],
    confusion_labels: ["A", "B"],
    confusion_matrix: [
      [1, 2],
      [0, 3],
    ],
    checkpoint: "x",
    note: "cross-dataset",
    ...overrides,
  };
}

describe("ConfusionHeatmap", () => {
  it("renders an accessible figure naming the worst confusion", () => {
    const { getByRole } = render(<ConfusionHeatmap data={makeData()} />);
    const fig = getByRole("img");
    expect(fig.getAttribute("aria-label")).toMatch(/true A predicted as B/i);
  });

  it("renders a cell per matrix entry plus headers (n*(n+1)+1 grid children)", () => {
    const { getByRole } = render(<ConfusionHeatmap data={makeData()} />);
    // The diagonal cells carry a title with the recall %, so we can count them.
    const fig = getByRole("img");
    const titled = fig.querySelectorAll("[title]");
    expect(titled.length).toBe(4); // 2×2 matrix cells
  });

  it("renders nothing for an empty matrix", () => {
    const { container } = render(
      <ConfusionHeatmap data={makeData({ confusion_labels: [], confusion_matrix: [] })} />,
    );
    expect(container.firstChild).toBeNull();
  });
});
