export interface ChannelViz {
  imageData: ImageData;
  label: string;
  range: [number, number];
}

export function tensorToChannelViz(
  tensor: Float32Array,
  size: number,
): [ChannelViz, ChannelViz, ChannelViz] {
  const pixels = size * size;
  if (tensor.length < 3 * pixels) {
    throw new Error(`Expected at least ${3 * pixels} values, got ${tensor.length}.`);
  }
  const labels: [string, string, string] = ["Red channel", "Green channel", "Blue channel"];
  const result: ChannelViz[] = [];
  for (let c = 0; c < 3; c++) {
    const offset = c * pixels;
    const slice = tensor.subarray(offset, offset + pixels);
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < pixels; i++) {
      const v = slice[i]!;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const data = new Uint8ClampedArray(pixels * 4);
    const range = max - min;
    const scale = range > 0 ? 255 / range : 0;
    for (let i = 0; i < pixels; i++) {
      const intensity = range > 0 ? Math.round((slice[i]! - min) * scale) : 128;
      const idx = i * 4;
      if (c === 0) { data[idx] = intensity; data[idx+1] = 0; data[idx+2] = 0; }
      else if (c === 1) { data[idx] = 0; data[idx+1] = intensity; data[idx+2] = 0; }
      else { data[idx] = 0; data[idx+1] = 0; data[idx+2] = intensity; }
      data[idx + 3] = 255;
    }
    const imageData = new ImageData(data, size, size);
    result.push({ imageData, label: labels[c]!, range: [min, max] });
  }
  return result as [ChannelViz, ChannelViz, ChannelViz];
}
