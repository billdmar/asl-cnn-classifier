import { describe, it, expect, vi } from "vitest";
import { captureVideoFrame, captureImageFrame } from "../pipeline-steps";

// Mock canvas context
const mockCtx = { drawImage: vi.fn() };
vi.stubGlobal("document", {
  createElement: vi.fn(() => ({
    width: 0,
    height: 0,
    getContext: () => mockCtx,
  })),
});

describe("captureVideoFrame", () => {
  it("creates a canvas matching video dimensions", () => {
    const video = { videoWidth: 640, videoHeight: 480 } as HTMLVideoElement;
    const frame = captureVideoFrame(video);
    expect(frame.width).toBe(640);
    expect(frame.height).toBe(480);
    expect(mockCtx.drawImage).toHaveBeenCalledWith(video, 0, 0, 640, 480);
  });
});

describe("captureImageFrame", () => {
  it("creates a canvas matching image dimensions", () => {
    mockCtx.drawImage.mockClear();
    const img = { naturalWidth: 320, naturalHeight: 240 } as HTMLImageElement;
    const frame = captureImageFrame(img);
    expect(frame.width).toBe(320);
    expect(frame.height).toBe(240);
    expect(mockCtx.drawImage).toHaveBeenCalledWith(img, 0, 0, 320, 240);
  });
});
