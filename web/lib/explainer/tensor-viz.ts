/**
 * Tensor channel visualization — converts a CHW normalized float32 tensor into
 * per-channel ImageData objects suitable for rendering on a canvas.
 *
 * Each channel is min-max rescaled to [0, 255] and rendered in its
 * representative color (R channel → red intensity, G → green, B → blue), making
 * the preprocessing transform visible to the user.
 */

/**
 * Create an ImageData instance, falling back to a plain object with the same
 * shape when the global `ImageData` constructor is unavailable (e.g. jsdom in
 * unit tests). The returned object is structurally compatible with
 * `CanvasRenderingContext2D.putImageData()`.
 */
function createImageData(
  data: Uint8ClampedArray,
  width: number,
  height: number,
): ImageData {
  if (typeof ImageData !== "undefined") {
    return new ImageData(data, width, height);
  }
  // Structural fallback for test environments missing the constructor.
  return { data, width, height, colorSpace: "srgb" } as ImageData;
}

/** A single channel's visualization data ready for canvas rendering. */
export interface ChannelViz {
  /** ImageData sized `size x size`, ready for `ctx.putImageData()`. */
  imageData: ImageData;
  /** Human-readable channel label. */
  label: string;
  /** The [min, max] range of the original normalized values in this channel. */
  range: [number, number];
}

/**
 * Convert a CHW normalized tensor (3 * size * size Float32Array) into three
 * per-channel ImageData objects suitable for `canvas.putImageData()`.
 *
 * Each channel is min-max rescaled to [0, 255] and rendered in its
 * representative color (R channel = red intensity, G = green, B = blue).
 *
 * @param tensor - Float32Array of length `3 * size * size` in CHW layout.
 * @param size - The spatial side length (defaults to 128).
 * @returns A tuple of three {@link ChannelViz} objects: [R, G, B].
 */
export function tensorToChannelViz(
  tensor: Float32Array,
  size: number,
): [ChannelViz, ChannelViz, ChannelViz] {
  const pixels = size * size;
  if (tensor.length < 3 * pixels) {
    throw new Error(
      `Expected at least ${3 * pixels} values for a 3x${size}x${size} tensor, got ${tensor.length}.`,
    );
  }

  const labels: [string, string, string] = [
    "Red channel",
    "Green channel",
    "Blue channel",
  ];

  const result: ChannelViz[] = [];

  for (let c = 0; c < 3; c++) {
    const offset = c * pixels;
    const slice = tensor.subarray(offset, offset + pixels);

    // Find the min/max for rescaling.
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < pixels; i++) {
      const v = slice[i]!;
      if (v < min) min = v;
      if (v > max) max = v;
    }

    // Build RGBA pixel data with the channel's representative color.
    const data = new Uint8ClampedArray(pixels * 4);
    const range = max - min;
    // Guard against constant-value channels (range=0 → all pixels at 128).
    const scale = range > 0 ? 255 / range : 0;

    for (let i = 0; i < pixels; i++) {
      const intensity = range > 0
        ? Math.round((slice[i]! - min) * scale)
        : 128;
      const idx = i * 4;
      // Write intensity to the channel's position only; others stay 0.
      if (c === 0) {
        data[idx] = intensity;       // R
        data[idx + 1] = 0;
        data[idx + 2] = 0;
      } else if (c === 1) {
        data[idx] = 0;
        data[idx + 1] = intensity;   // G
        data[idx + 2] = 0;
      } else {
        data[idx] = 0;
        data[idx + 1] = 0;
        data[idx + 2] = intensity;   // B
      }
      data[idx + 3] = 255;          // Alpha
    }

    // Use the data-first constructor form which is available in all environments
    // where ImageData exists. In jsdom (test) it may not be defined globally, so
    // the test must polyfill or the caller must provide a canvas-backed env.
    const imageData = createImageData(data, size, size);

    result.push({
      imageData,
      label: labels[c]!,
      range: [min, max],
    });
  }

  return result as [ChannelViz, ChannelViz, ChannelViz];
}
