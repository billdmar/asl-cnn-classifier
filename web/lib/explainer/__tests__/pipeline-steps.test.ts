import { describe, it, expect, vi, beforeEach } from "vitest";
import { captureVideoFrame, captureImageFrame } from "../pipeline-steps";

/**
 * Minimal mock of the 2D canvas context — tracks drawImage calls and provides
 * a fake getImageData for tests that need pixel data.
 */
function createMockCtx() {
  return {
    drawImage: vi.fn(),
    getImageData: vi.fn(() => ({
      data: new Uint8ClampedArray(4),
      width: 1,
      height: 1,
    })),
  };
}

/** Minimal mock canvas that records width/height assignments. */
function createMockCanvas(ctx: ReturnType<typeof createMockCtx>) {
  return {
    width: 0,
    height: 0,
    getContext: vi.fn(() => ctx),
  };
}

describe("captureVideoFrame", () => {
  let mockCtx: ReturnType<typeof createMockCtx>;
  let mockCanvas: ReturnType<typeof createMockCanvas>;

  beforeEach(() => {
    mockCtx = createMockCtx();
    mockCanvas = createMockCanvas(mockCtx);
    vi.spyOn(document, "createElement").mockImplementation((tag: string) => {
      if (tag === "canvas") return mockCanvas as unknown as HTMLCanvasElement;
      return document.createElement(tag);
    });
  });

  it("creates a canvas matching video dimensions", () => {
    const video = {
      videoWidth: 640,
      videoHeight: 480,
    } as HTMLVideoElement;

    const frame = captureVideoFrame(video);

    expect(frame.width).toBe(640);
    expect(frame.height).toBe(480);
    expect(mockCanvas.width).toBe(640);
    expect(mockCanvas.height).toBe(480);
  });

  it("draws the video onto the canvas", () => {
    const video = {
      videoWidth: 320,
      videoHeight: 240,
    } as HTMLVideoElement;

    captureVideoFrame(video);

    expect(mockCtx.drawImage).toHaveBeenCalledWith(video, 0, 0, 320, 240);
  });

  it("returns a FrozenFrame with the canvas as source", () => {
    const video = {
      videoWidth: 1920,
      videoHeight: 1080,
    } as HTMLVideoElement;

    const frame = captureVideoFrame(video);

    expect(frame.source).toBe(mockCanvas);
    expect(frame.width).toBe(1920);
    expect(frame.height).toBe(1080);
  });
});

describe("captureImageFrame", () => {
  let mockCtx: ReturnType<typeof createMockCtx>;
  let mockCanvas: ReturnType<typeof createMockCanvas>;

  beforeEach(() => {
    mockCtx = createMockCtx();
    mockCanvas = createMockCanvas(mockCtx);
    vi.spyOn(document, "createElement").mockImplementation((tag: string) => {
      if (tag === "canvas") return mockCanvas as unknown as HTMLCanvasElement;
      return document.createElement(tag);
    });
  });

  it("creates a canvas matching image natural dimensions", () => {
    const img = {
      naturalWidth: 800,
      naturalHeight: 600,
    } as HTMLImageElement;

    const frame = captureImageFrame(img);

    expect(frame.width).toBe(800);
    expect(frame.height).toBe(600);
    expect(mockCanvas.width).toBe(800);
    expect(mockCanvas.height).toBe(600);
  });

  it("draws the image onto the canvas", () => {
    const img = {
      naturalWidth: 1024,
      naturalHeight: 768,
    } as HTMLImageElement;

    captureImageFrame(img);

    expect(mockCtx.drawImage).toHaveBeenCalledWith(img, 0, 0, 1024, 768);
  });

  it("returns a FrozenFrame with the canvas as source", () => {
    const img = {
      naturalWidth: 256,
      naturalHeight: 256,
    } as HTMLImageElement;

    const frame = captureImageFrame(img);

    expect(frame.source).toBe(mockCanvas);
    expect(frame.width).toBe(256);
    expect(frame.height).toBe(256);
  });
});
