/**
 * Browser-only helper to download the current hand crop as a labeled PNG.
 *
 * Used by the webcam panel's "Build a test set" tool so a user can collect
 * labeled real-world frames (drop them into `data/realworld_eval/<CLASS>/`
 * later). Touches `document` / `URL` only, so it must be called from event
 * handlers, never during render or on the server.
 */

/** A–Z, the model's class labels, for the capture letter selector. */
export const CAPTURE_LETTERS: readonly string[] = Array.from({ length: 26 }, (_, i) =>
  String.fromCharCode(65 + i),
);

/** Filename-safe timestamp like `2026-06-24T12-30-05-123Z`. */
function timestampSlug(): string {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

/**
 * Encode a canvas to PNG and trigger a client-side download named
 * `<LETTER>_<timestamp>.png`. The temporary object URL is revoked after the
 * click so we don't leak blobs.
 *
 * @param canvas - The cropped 128x128 hand canvas being classified.
 * @param letter - The label the user selected for this frame.
 */
export function downloadCanvasAsPng(canvas: HTMLCanvasElement, letter: string): void {
  canvas.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${letter}_${timestampSlug()}.png`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }, "image/png");
}
