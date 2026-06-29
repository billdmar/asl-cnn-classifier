import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { ResultCard } from "@/components/result-card";
import type { SharedResult } from "@/lib/share-link";

const SAMPLE: SharedResult = {
  letter: "A",
  topk: [
    ["A", 0.99],
    ["S", 0.005],
  ],
  t: 1_700_000_000_000,
  v: 1,
};

describe("ResultCard", () => {
  it("renders the predicted letter prominently", () => {
    render(<ResultCard result={SAMPLE} />);
    // "A" appears both as the big letter and as a top-k bar label, so just
    // assert at least one is present.
    expect(screen.getAllByText("A").length).toBeGreaterThan(0);
  });

  it("renders top-k percentages", () => {
    render(<ResultCard result={SAMPLE} />);
    expect(screen.getByText("99.0%")).toBeInTheDocument();
  });

  it("links back to the live demo", () => {
    render(<ResultCard result={SAMPLE} />);
    const link = screen.getByRole("link", { name: /try it yourself/i });
    expect(link).toHaveAttribute("href", "/");
  });

  it("handles an empty top-k list without crashing", () => {
    render(<ResultCard result={{ letter: "Z", topk: [], t: 0, v: 1 }} />);
    expect(screen.getByText("Z")).toBeInTheDocument();
  });
});
