import { describe, it, expect } from "vitest";
import { tensorToChannelViz } from "../tensor-viz";

describe("tensorToChannelViz", () => {
  const size = 4;
  const pixels = size * size; // 16

  /**
   * Build a known tensor: 3 channels x 4x4 pixels = 48 values.
   * - R channel: linearly spaced [0.0, 1.0, 2.0, ..., 15.0]
   * - G channel: constant 0.5
   * - B channel: linearly spaced [-1.0, -0.9, -0.8, ..., 0.5]
   */
  function makeTensor(): Float32Array {
    const tensor = new Float32Array(3 * pixels);
    // R channel
    for (let i = 0; i < pixels; i++) tensor[i] = i;
    // G channel (constant)
    for (let i = 0; i < pixels; i++) tensor[pixels + i] = 0.5;
    // B channel (negative range)
    for (let i = 0; i < pixels; i++) tensor[2 * pixels + i] = -1.0 + i * 0.1;
    return tensor;
  }

  it("returns three ChannelViz objects with correct labels", () => {
    const tensor = makeTensor();
    const [r, g, b] = tensorToChannelViz(tensor, size);

    expect(r.label).toBe("Red channel");
    expect(g.label).toBe("Green channel");
    expect(b.label).toBe("Blue channel");
  });

  it("reports correct min/max ranges for each channel", () => {
    const tensor = makeTensor();
    const [r, g, b] = tensorToChannelViz(tensor, size);

    // R: min=0, max=15
    expect(r.range[0]).toBeCloseTo(0, 6);
    expect(r.range[1]).toBeCloseTo(15, 6);

    // G: constant 0.5
    expect(g.range[0]).toBeCloseTo(0.5, 6);
    expect(g.range[1]).toBeCloseTo(0.5, 6);

    // B: min=-1.0, max = -1.0 + 15*0.1 = 0.5
    expect(b.range[0]).toBeCloseTo(-1.0, 6);
    expect(b.range[1]).toBeCloseTo(0.5, 6);
  });

  it("produces correctly-sized ImageData for each channel", () => {
    const tensor = makeTensor();
    const [r, g, b] = tensorToChannelViz(tensor, size);

    expect(r.imageData.width).toBe(size);
    expect(r.imageData.height).toBe(size);
    expect(r.imageData.data.length).toBe(pixels * 4);

    expect(g.imageData.width).toBe(size);
    expect(g.imageData.height).toBe(size);

    expect(b.imageData.width).toBe(size);
    expect(b.imageData.height).toBe(size);
  });

  it("R channel min-max scales pixels correctly", () => {
    const tensor = makeTensor();
    const [r] = tensorToChannelViz(tensor, size);
    const data = r.imageData.data;

    // R channel values are [0..15]. After min-max scaling:
    // pixel 0: (0-0)/(15-0)*255 = 0
    // pixel 15: (15-0)/(15-0)*255 = 255
    // pixel 8: (8-0)/(15-0)*255 = 136 (rounded)

    // First pixel: R=0, G=0, B=0, A=255
    expect(data[0]).toBe(0);
    expect(data[1]).toBe(0);
    expect(data[2]).toBe(0);
    expect(data[3]).toBe(255);

    // Last pixel (index 15): R=255, G=0, B=0, A=255
    const lastIdx = 15 * 4;
    expect(data[lastIdx]).toBe(255);
    expect(data[lastIdx + 1]).toBe(0);
    expect(data[lastIdx + 2]).toBe(0);
    expect(data[lastIdx + 3]).toBe(255);

    // Middle pixel (index 8): R=round(8/15*255)=136
    const midIdx = 8 * 4;
    expect(data[midIdx]).toBe(Math.round((8 / 15) * 255));
    expect(data[midIdx + 1]).toBe(0);
    expect(data[midIdx + 2]).toBe(0);
  });

  it("G channel with constant value renders as uniform intensity 128", () => {
    const tensor = makeTensor();
    const [, g] = tensorToChannelViz(tensor, size);
    const data = g.imageData.data;

    // All G channel values are 0.5 (constant) → range=0 → fallback to 128.
    for (let i = 0; i < pixels; i++) {
      const idx = i * 4;
      expect(data[idx]).toBe(0);     // R=0
      expect(data[idx + 1]).toBe(128); // G=128 (constant fallback)
      expect(data[idx + 2]).toBe(0); // B=0
      expect(data[idx + 3]).toBe(255); // A=255
    }
  });

  it("B channel renders intensity only in the blue position", () => {
    const tensor = makeTensor();
    const [, , b] = tensorToChannelViz(tensor, size);
    const data = b.imageData.data;

    // First pixel: B channel value = -1.0 (the min) → intensity = 0
    expect(data[0]).toBe(0);   // R
    expect(data[1]).toBe(0);   // G
    expect(data[2]).toBe(0);   // B = min → 0
    expect(data[3]).toBe(255); // A

    // Last pixel: B channel value = -1.0 + 15*0.1 = 0.5 (the max) → intensity = 255
    const lastIdx = 15 * 4;
    expect(data[lastIdx]).toBe(0);
    expect(data[lastIdx + 1]).toBe(0);
    expect(data[lastIdx + 2]).toBe(255);
    expect(data[lastIdx + 3]).toBe(255);
  });

  it("throws when tensor is too short", () => {
    const short = new Float32Array(10);
    expect(() => tensorToChannelViz(short, size)).toThrow(/Expected at least/);
  });
});
