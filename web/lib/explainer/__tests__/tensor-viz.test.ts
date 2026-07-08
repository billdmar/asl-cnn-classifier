import { describe, it, expect, beforeAll } from "vitest";
import { tensorToChannelViz } from "../tensor-viz";

// jsdom does not ship ImageData; provide a minimal polyfill for tests.
beforeAll(() => {
  if (typeof globalThis.ImageData === "undefined") {
    (globalThis as unknown as Record<string, unknown>).ImageData = class ImageData {
      data: Uint8ClampedArray;
      width: number;
      height: number;
      constructor(data: Uint8ClampedArray, width: number, height: number) {
        this.data = data;
        this.width = width;
        this.height = height;
      }
    };
  }
});

describe("tensorToChannelViz", () => {
  const size = 4;
  const pixels = size * size;

  it("returns three channel visualizations with correct labels", () => {
    const tensor = new Float32Array(3 * pixels).fill(0);
    const [r, g, b] = tensorToChannelViz(tensor, size);
    expect(r.label).toBe("Red channel");
    expect(g.label).toBe("Green channel");
    expect(b.label).toBe("Blue channel");
  });

  it("produces ImageData of correct dimensions", () => {
    const tensor = new Float32Array(3 * pixels).fill(0.5);
    const [r] = tensorToChannelViz(tensor, size);
    expect(r.imageData.width).toBe(size);
    expect(r.imageData.height).toBe(size);
    expect(r.imageData.data.length).toBe(pixels * 4);
  });

  it("rescales values to 0-255 range for non-constant channels", () => {
    const tensor = new Float32Array(3 * pixels);
    for (let i = 0; i < pixels; i++) tensor[i] = i / (pixels - 1);
    tensor.fill(0.5, pixels, 3 * pixels);
    const [r] = tensorToChannelViz(tensor, size);
    expect(r.imageData.data[0]).toBe(0);
    expect(r.imageData.data[(pixels - 1) * 4]).toBe(255);
    expect(r.range[0]).toBeCloseTo(0);
    expect(r.range[1]).toBeCloseTo(1);
  });

  it("uses 128 for constant-value channels", () => {
    const tensor = new Float32Array(3 * pixels).fill(0.5);
    const [r] = tensorToChannelViz(tensor, size);
    expect(r.imageData.data[0]).toBe(128);
  });

  it("throws on short tensor", () => {
    expect(() => tensorToChannelViz(new Float32Array(10), size)).toThrow();
  });

  it("colors only the correct channel", () => {
    const tensor = new Float32Array(3 * pixels);
    for (let i = 0; i < pixels; i++) tensor[i] = i / (pixels - 1);
    for (let i = pixels; i < 2 * pixels; i++) tensor[i] = i / (pixels - 1);
    for (let i = 2 * pixels; i < 3 * pixels; i++) tensor[i] = i / (pixels - 1);
    const [r, g, b] = tensorToChannelViz(tensor, size);
    // R channel: only red bytes non-zero
    expect(r.imageData.data[1]).toBe(0); // G byte
    expect(r.imageData.data[2]).toBe(0); // B byte
    // G channel: only green bytes non-zero
    expect(g.imageData.data[0]).toBe(0); // R byte
    expect(g.imageData.data[2]).toBe(0); // B byte
    // B channel: only blue bytes non-zero
    expect(b.imageData.data[0]).toBe(0); // R byte
    expect(b.imageData.data[1]).toBe(0); // G byte
  });
});
